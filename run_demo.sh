# cd {this_repo}/ManiSkill2_real2sim

python ManiSkill2_real2sim/mani_skill2_real2sim/examples/demo_manual_control_custom_envs.py \
    -e GraspSingleOpenedCokeCanInScene-v0 \
    -c arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner \
    -o rgbd \
    --enable-sapien-viewer \
    prepackaged_config @True \
    robot google_robot_static