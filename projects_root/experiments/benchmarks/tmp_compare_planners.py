def franka_k_arm_centralized_mpc():
    num_arms = auto_detect_num_arms(robot_config_path)
    arm_targets = create_arm_targets(num_arms, my_world, robot_cfg)
    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)
    articulation_controller = robot.get_articulation_controller()
    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        use_cuda_graph=False,
        use_cuda_graph_metrics=True,
        use_cuda_graph_full_step=False,
        self_collision_check=True,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        use_mppi=True,
        use_lbfgs=False,
        use_es=False,
        store_rollouts=True,
        step_dt=0.02,
        override_particle_file=particle_config_path,
        num_arms=num_arms  # Pass the auto-detected number of arms
    )


def curobo_multi_arm_reacher():
    """
    centralized multi arm reacher
    (demonstrated on 2 arms)
    """
    ...
    # init

    # multi armed single urdf
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]
    
    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)
    articulation_controller = robot.get_articulation_controller()
    world_cfg = ...
    motion_gen_config = MotionGenConfig.load_from_robot_config ... # solver config
    motion_gen = MotionGen(motion_gen_config) # ... solver  
    plan_config = MotionGenPlanConfig ()# ... planner config..
    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")


    # create targets:
    names = []
    for i in link_names:
        if i != ee_link_name:
            k_pose = np.ravel(link_retract_pose[i].to_list())
            target_links[i] = cuboid.VisualCuboid(
                "/World/target_" + i,
                ...
            )
            names.append("/World/target_" + i)


    # loop:
    while simulation_app.is_running():
        my_world.step(render=True)
        # if any of the targets moved:
        # add link poses:
        target_poses_from_sim = {} # 
        for target_id in range(len(targets)):
            c_p, c_rot = target.get_world_pose()
            i target has moved: # (c_p, c_rot difference from previous step)
                target_poses_from_sim[target] = Pose(
                    position=tensor_args.to_device(c_p),
                    quaternion=tensor_args.to_device(c_rot),
                )
        result = motion_gen.plan_single(
            cu_js.unsqueeze(0), ik_goal, plan_config.clone(), link_poses=link_poses
        )