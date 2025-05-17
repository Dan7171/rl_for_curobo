import abc

class Iprim(abc.ABC):
    
    def __init__(self,*args,**kwargs):
        pass

    @abc.abstractmethod
    def spawn_prim(self,*args,**kwargs):
        """Spawn the prim in the stage"""
        pass
    
    # @abc.abstractmethod
    # def set_path(self,*args,**kwargs):
    #     """Set the path of the prim in the stage"""
    #     pass
    
    @abc.abstractmethod 
    def get_path(self,*args,**kwargs):
        """Get the path of the prim in the stage"""
        pass
    
    
    @abc.abstractmethod
    def enable_collision(self,*args,**kwargs):
        """Enable collision for the prim (in the meaning of touching other prims)"""
        pass

    @abc.abstractmethod
    def disable_collision(self,*args,**kwargs):
        """Disable collision for the prim (in the meaning of touching other prims)"""
        pass
    
    
    @abc.abstractmethod
    def enable_gravity(self,*args,**kwargs):
        pass
    
    @abc.abstractmethod
    def disable_gravity(self,*args,**kwargs):
        pass
    
    @abc.abstractmethod
    def get_pose(self,*args,**kwargs):
        """Get the pose of the prim in the world"""
        pass
    
    @abc.abstractmethod
    def get_velocity(self,*args,**kwargs):
        pass

    # @abc.abstractmethod
    # def get_acceleration(self,*args,**kwargs):
    #     pass
    
    @abc.abstractmethod
    def set_velocity(self,*args,**kwargs):
        pass
    
    # @abc.abstractmethod
    # def set_acceleration(self,*args,**kwargs):
    #     pass
    
    @abc.abstractmethod
    def set_position(self,*args,**kwargs):
        """Set the position of the prim in the world"""
        pass
    
    @abc.abstractmethod
    def set_rotation(self,*args,**kwargs):
        """Set the rotation of the prim in the world"""
        pass

    
    
    