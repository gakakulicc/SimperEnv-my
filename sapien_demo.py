import sapien.core as sapien
import numpy as np
import cv2

# 初始化引擎和渲染器
engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)

# 创建场景
scene_config = sapien.SceneConfig()
scene = engine.create_scene(scene_config)
scene.set_timestep(1 / 240.0)
scene.add_ground(0)

# 设置环境光和方向光
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

# 创建关节连接
articulation_builder = scene.create_articulation_builder()

# 创建底座（根链接）
root_builder = articulation_builder.create_link_builder()
root_builder.add_box_collision(half_size=[0.1, 0.1, 0.05])
root_builder.add_box_visual(half_size=[0.1, 0.1, 0.05], color=[0.8, 0.8, 0.8])
root_builder.set_name('base')

# 创建第一个关节（旋转关节）
link1_builder = articulation_builder.create_link_builder(root_builder)
link1_builder.add_box_collision(half_size=[0.05, 0.05, 0.05])
link1_builder.add_box_visual(half_size=[0.05, 0.05, 0.05], color=[0.8, 0.8, 0.8])
link1_builder.set_name('link1')
link1_builder.set_joint_name('joint1')  # 设置关节名称
link1_builder.set_joint_properties(
    joint_type='revolute',  # 关节类型
    limits=[[-np.pi, np.pi]],  # 关节运动范围
    pose_in_parent=sapien.Pose([0, 0, 0.1]),  # 父链接到关节的相对位置
    pose_in_child=sapien.Pose([0, 0, 0]),  # 子链接到关节的相对位置
    friction=0.0,  # 关节摩擦
    damping=0.0  # 关节阻尼
)

# 创建第二个关节（旋转关节）
link2_builder = articulation_builder.create_link_builder(link1_builder)
link2_builder.add_box_collision(half_size=[0.05, 0.05, 0.05])
link2_builder.add_box_visual(half_size=[0.05, 0.05, 0.05], color=[0.8, 0.8, 0.8])
link2_builder.set_name('link2')
link2_builder.set_joint_name('joint2')  # 设置关节名称
link2_builder.set_joint_properties(
    joint_type='revolute',  # 关节类型
    limits=[[-np.pi, np.pi]],  # 关节运动范围
    pose_in_parent=sapien.Pose([0, 0, 0.1]),  # 父链接到关节的相对位置
    pose_in_child=sapien.Pose([0, 0, 0]),  # 子链接到关节的相对位置
    friction=0.0,  # 关节摩擦
    damping=0.0  # 关节阻尼
)

# 构建关节
articulation = articulation_builder.build()
articulation.set_root_pose(sapien.Pose([0, 0, 0.05]))

# 获取第一个关节
joint1 = articulation.get_joints()[0]
joint1.set_drive_velocity_target([1.0])  # 设置关节速度目标，传递单个浮点数

# 设置视频输出参数
frame_width = 640
frame_height = 480
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码器
out = cv2.VideoWriter('simulation_output.mp4', fourcc, 30.0, (frame_width, frame_height))

# 创建相机
camera = scene.add_camera(
    name="camera",
    width=frame_width,
    height=frame_height,
    fovy=np.deg2rad(35),  # 视场角
    near=0.1,
    far=100
)

# 设置相机位置和朝向
camera.set_pose(sapien.Pose(p=[0, -1, 0.5], q=[1, 0, 0, 0]))  # 使用单位四元数表示无旋转

# 渲染并保存视频
for _ in range(240):  # 渲染 240 帧
    scene.step()
    scene.update_render()
    camera.take_picture()
    image = camera.get_color_rgba()  # 获取渲染图像
    frame = (image[:, :, :3] * 255).astype(np.uint8)  # 转换为 uint8 格式
    out.write(frame)

out.release()