import inspect
import functools
from collections.abc import Callable
import numpy as np
import logging
import copy

from impact_t_beamline import ImpactTBeamline, try_except_timeout, block_negative_velocity
from beamline_configuration import BeamlineConfiguration
from impact_input import ImpactIN

logger = logging.getLogger(__name__)

def get_beamline_instance(args):
    return [arg for arg in args if isinstance(arg, ImpactTBeamline)][0]

def no_timeout(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except subprocess.TimeoutExpired:
            logger.warning(f'Timeout Occured')
            return -1
        except subprocess.CalledProcessError as e:
            print('Command failed with error:', e.returncode, e.output)  
            raise
    wrapper.__signature__ = inspect.signature(func)
    return wrapper    

def check_sc_limit_emission(zero_crossing_phase=-48.61939264802021,sc_factor=1.1,radius_name='distgen__r_dist:truncation_radius:value'):
    def check_sc_limit_emission_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            temp_beamline = get_beamline_instance(args)
            
            gun_amp = temp_beamline.settings['GunAmp']
            gun_phase = temp_beamline.settings['GunPhase']
            rRMS = temp_beamline.settings[radius_name]
            Q_TOT = temp_beamline.settings['Qtotal']
            
            eF_sc = (Q_TOT/8.85e-12/np.pi/rRMS**2)*1e-6 #MV/m
            eF_gun = gun_amp*np.sin(np.deg2rad(gun_phase)-np.deg2rad(zero_crossing_phase)) #MV/m

            logger.info(f'Space Charge Limited Emission Test - E_gun: {eF_gun} MV/m - E_SC: {eF_sc} MV/m')
            if eF_gun > sc_factor*eF_sc:
                logger.info(f'Space Charge Limited Emission Test: Pass')
                return func(*args, **kwargs)
            else:
                logger.info(f'Space Charge Limited Emission Test: Fail')
                return False
        wrapper.__signature__ = inspect.signature(func)
        return wrapper
    return check_sc_limit_emission_decorator

def no_negative_velocity(func):
    def wrapper(*args, **kwargs):  
        temp_beamline = get_beamline_instance(args)
        z_df = temp_beamline.getFort(fort_num=26)
        
        if np.any(z_df.avgPz<0):
            logger.info(f'Negative Velocity Test: Fail')
            return False
        else:
            logger.info(f'Negative Velocity Test: Pass')
            return func(*args, **kwargs)
    wrapper.__signature__ = inspect.signature(func)
    return wrapper

def monotonic_energy_gain(z_pos,tol):
    def monotonic_energy_gain_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            temp_beamline = get_beamline_instance(args)
            ref_df = temp_beamline.getFort_z_pos(18,z_pos_list=sorted(z_pos))

            ke_vals = ref_df.KE.values
            logger.info(f'Monotonic Energy Test: {ke_vals} MeV')
            if np.all(np.diff(ke_vals) > -tol):
                logger.info(f'Monotonic Energy Test: Pass')
                return func(*args, **kwargs)
            else:
                logger.info(f'Monotonic Energy Test: Fail')
                return False
        wrapper.__signature__ = inspect.signature(func)
        return wrapper
    return monotonic_energy_gain_decorator

def final_energy(final_ke,tol):
    def final_energy_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Filter positional arguments based on their type            
            temp_beamline = get_beamline_instance(args)
            ref_df = temp_beamline.getFort(fort_num=18)
            
            logger.info(f'Final Energy Test - final energy: {ref_df.KE.values[-1]} MeV - target energy: {final_ke} MeV')
            if np.abs(ref_df.KE.values[-1] - final_ke) < tol :
                return func(*args, **kwargs)
            else:
                return False
        wrapper.__signature__ = inspect.signature(func)
        return wrapper
    return final_energy_decorator

class ManageBeamline:
    def __init__(self, beamline: ImpactTBeamline,beamline_config: BeamlineConfiguration, ):
        self.beamline = beamline
        self.beamline_config = beamline_config
    
    def update(self, val_dict: dict):
        # update each value in the original settings file
        for key,val in val_dict.items():
            self.beamline_config.settings[key]['input']['value'] = val
        
        # generate values from beamline_config for input into beamline
        output_dict = self.beamline_config.gen()
        self.beamline.settings = output_dict
        
    def get(self, key: str):
        # update each value in the original settings file
        return self.beamline_config.settings[key]['input']['value']
    
    def __copy__(self):
        # Return a shallow copy of the object
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        # Return a deep copy of the object
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__ = copy.deepcopy(self.__dict__, memo)
        return result
    
    @try_except_timeout
    def run(self):
        self.beamline.run()
        return 1
    
    def eval_beamline(self,func: Callable):
        return func(self.beamline)
