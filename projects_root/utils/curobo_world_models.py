
from curobo.geom.types import WorldConfig
 


def update_world_model(current:WorldConfig, 
                       other:WorldConfig,
                       conflict_resolution_strategy:str='replace'):
    """
    Update the current world model with the other world model.

    Args:
        current (WorldConfig): The current world model to be updated.
        other (WorldConfig): The other world model to update with.
        conflict_resolution_strategy (str, optional): The strategy to resolve conflicts. Defaults to 'replace'.
            - 'replace': Replace the current object with the new one.
            - 'keep': Keep the current object.
    """

    # object_types = ['blox', 'capsule', 'cuboid', 'cylinder', 'mesh', 'sphere']
    models = [current, other]
    objects = []
    object_names = []
    for model in models:
        d_objects = {
            'blox':model.blox,
            'capsule':model.capsule,
            'cuboid':model.cuboid,
            'cylinder':model.cylinder,
            'mesh':model.mesh,
            'sphere':model.sphere
        }
        objects.append(d_objects)

        d_names = {}
        for object_type in d_objects.keys():
            d_names[object_type] = []
            for object in d_objects[object_type]:
                d_names[object_type].append(object.name)
        object_names.append(d_names)
    
    current_object_names = object_names[0]
    other_object_names = object_names[1]
    current_objects = objects[0]
    other_objects = objects[1]
    
    # objects in other that are not in current - add them
    
    for object_type in current_objects.keys():
        n_in_type_current = len(current_objects[object_type])
        for i in range(len(other_object_names[object_type])):
            name_other = other_object_names[object_type][i]
            found_match_under_object_type = False
            found_match_under_general_objects = False
            for j in range(n_in_type_current):
                name_current = current_object_names[object_type][j]
                if name_current == name_other:
                    found_match_under_object_type = True
                    if conflict_resolution_strategy == 'replace': # replace the current with the new one
                        # replace at the specific object type
                        current_objects[object_type][j] = other_objects[object_type][i]
                        # replace at the general objects list
                        found_match_under_general_objects = False
                        for k,o in enumerate(current.objects):
                            if o.name == name_current:
                                current.objects[k] = other_objects[object_type][i]
                                found_match_under_general_objects = True
                                print(f'Replacing {name_current} with {name_other} under the general objects list')
                                
                                break
                        if not found_match_under_general_objects:
                            raise ValueError(f"Could not find match for {name_current} under the general objects list")
                    elif conflict_resolution_strategy == 'keep': # keep the current obstacle
                        pass # do nothing
                    else:
                        raise ValueError(f"Conflict resolution strategy {conflict_resolution_strategy} not supported")
            
            if found_match_under_object_type and not found_match_under_general_objects:
                raise ValueError("If match found under the object type, must also find match under the general objects list")
            
            else: # add the new object
                current.add_obstacle(other_objects[object_type][i])
        


            