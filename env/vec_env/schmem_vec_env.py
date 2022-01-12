"""
An interface for asynchronous vectorized environments.
"""

import multiprocessing as mp
import numpy as np
from .vec_env import VecEnv, CloudpickleWrapper, clear_mpi_env_vars
import ctypes

from .util import dict_to_obs, obs_space_info, obs_to_dict

_NP_TO_CT = {np.float32: ctypes.c_float,
             np.int32: ctypes.c_int32,
             np.int8: ctypes.c_int8,
             np.uint8: ctypes.c_char,
             np.bool: ctypes.c_bool}


class ShmemVecEnv(VecEnv):
    """
    Optimized version of SubprocVecEnv that uses shared variables to communicate observations.
    """

    def __init__(self, env_fns, spaces=None, context='spawn', penalty=False):
        """
        If you don't specify observation_space, we'll have to create a dummy
        environment to get it.
        """
        ctx = mp.get_context(context)
        if spaces:
            observation_space = spaces
        VecEnv.__init__(self, len(env_fns))
        self.obs_keys, self.obs_shapes, self.obs_dtypes = obs_space_info(observation_space)
        self.obs_bufs = [
            {k: ctx.Array(_NP_TO_CT[self.obs_dtypes[k].type], int(np.prod(self.obs_shapes[k]))) for k in self.obs_keys}
            for _ in env_fns]
        self.parent_pipes = []
        self.procs = []
        self.number = 0
        with clear_mpi_env_vars():
            for env_fn, obs_buf in zip(env_fns, self.obs_bufs):
                self.number += 1
                ctrl_index = int(self.number % 2)
                wrapped_fn = CloudpickleWrapper(env_fn)
                parent_pipe, child_pipe = ctx.Pipe()
                proc = ctx.Process(target=_subproc_worker,
                            args=(child_pipe, parent_pipe, wrapped_fn, obs_buf, self.obs_shapes, self.obs_dtypes, self.obs_keys, ctrl_index, penalty))
                proc.daemon = True
                self.procs.append(proc)
                self.parent_pipes.append(parent_pipe)
                proc.start()
                child_pipe.close()
        self.waiting_step = False
        self.viewer = None

    def reset(self):
        if self.waiting_step:
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('reset', None))
        return self._decode_obses([pipe.recv() for pipe in self.parent_pipes])

    def step_async(self, actions):
        assert len(actions) == len(self.parent_pipes)
        for pipe, act in zip(self.parent_pipes, actions):
            pipe.send(('step', act))
        self.waiting_step = True

    def step_wait(self):
        outs = [pipe.recv() for pipe in self.parent_pipes]
        self.waiting_step = False
        obs, rews, dones, infos = zip(*outs)
        return self._decode_obses(obs), np.array(rews), np.array(dones), infos

    def close_extras(self):
        if self.waiting_step:
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('close', None))
        for pipe in self.parent_pipes:
            pipe.recv()
            pipe.close()
        for proc in self.procs:
            proc.join()

    def get_images(self, mode='human'):
        for pipe in self.parent_pipes:
            pipe.send(('render', None))
        return [pipe.recv() for pipe in self.parent_pipes]

    def _decode_obses(self, obs):
        result = {}
        for k in self.obs_keys:

            bufs = [b[k] for b in self.obs_bufs]
            o = [np.frombuffer(b.get_obj(), dtype=self.obs_dtypes[k]).reshape(self.obs_shapes[k]) for b in bufs]
            result[k] = np.array(o)
        return dict_to_obs(result)


def _subproc_worker(pipe, parent_pipe, env_fn_wrapper, obs_bufs, obs_shapes, obs_dtypes, keys, ctrl_index, penalty=False):
    """
    Control a single environment instance using IPC and
    shared memory.
    """
    ctrl_index = ctrl_index # the index  of agent that we are controlling 
    def _write_obs(maybe_dict_obs, obs_sequence_dict):
        if ctrl_index == 0:
            ob_ctrl = maybe_dict_obs[0]['obs']
            ob_oppo = maybe_dict_obs[1]['obs']
        if ctrl_index == 1:
            ob_ctrl = maybe_dict_obs[1]['obs']
            ob_oppo = maybe_dict_obs[0]['obs']
        ob_ctrl = ob_ctrl.reshape(1, *ob_ctrl.shape)
        ob_oppo = ob_oppo.reshape(1, *ob_oppo.shape)
        if obs_sequence_dict is not None:
            obs_sequence1 = np.concatenate((np.delete(obs_sequence_dict['0'], 0, axis=0), ob_ctrl), axis=0)
            obs_sequence2 = np.concatenate((np.delete(obs_sequence_dict['1'], 0, axis=0), ob_oppo), axis=0)
            # a = obs_sequence_dict['0'][1]
            # b = obs_sequence1[0]
            # assert((a==b).all())
            # assert((ob_ctrl==obs_sequence1[-1]).all())
        else:
            obs_sequence1 = np.repeat(ob_ctrl, 4, axis=0)
            obs_sequence2 = np.repeat(ob_oppo, 4, axis=0)
        dict = {"0": obs_sequence1, "1": obs_sequence2}
        flatdict = obs_to_dict(dict)
        for k in keys:
            dst = obs_bufs[k].get_obj()
            dst_np = np.frombuffer(dst, dtype=obs_dtypes[k]).reshape(obs_shapes[k])  # pylint: disable=W0212
            np.copyto(dst_np, flatdict[k])
        return dict

    env = env_fn_wrapper.x()
    parent_pipe.close()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == 'reset':
                # ctrl_index = np.random.randint(0,2)
                wait_is_fatigue = 0
                last_obs_sequence_dict = _write_obs(env.reset(True), None)
                pipe.send(None)
            elif cmd == 'step':
                if ctrl_index == 1:
                    data.reverse() # put the action in the right place
                obs, reward, done, _, info = env.step(data)
                if penalty and not done:
                    if env.env_core.agent_list[ctrl_index].is_fatigue:
                        wait_is_fatigue +=1
                        if wait_is_fatigue == 3:
                            done = True
                if ctrl_index == 1:
                    reward.reverse() 
                if done:
                    # ctrl_index = np.random.randint(0,2)
                    obs = env.reset(True)
                    wait_is_fatigue = 0
                    last_obs_sequence_dict = None
                obs_sequence_dict = _write_obs(obs, last_obs_sequence_dict)
                last_obs_sequence_dict = obs_sequence_dict
                pipe.send((1, reward, done, info))
            elif cmd == 'render':
                pipe.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                pipe.send(None)
                break
            else:
                raise RuntimeError('Got unrecognized cmd %s' % cmd)
    except KeyboardInterrupt:
        print('ShmemVecEnv worker: got KeyboardInterrupt')
    # finally:
    #     env.close()
