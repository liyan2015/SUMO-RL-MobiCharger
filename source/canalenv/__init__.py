from gym.envs.registration import register

register(
   	id='SumoEnv-v0',
   	entry_point='canalenv.envs:SumoEnv',
    kwargs={
        'label':'default', 
        'gui_f':False
    },
)