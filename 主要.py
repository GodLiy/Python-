import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math
class FingerControlledSnakeGame:
    def __init__(self):
        # 初始化Mediapipe手部检测
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            static_image_mode=False  # 设置为False以消除警告
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # 游戏参数
        self.game_width = 0
        self.game_height = 0
        self.game_x_offset = 50  # 游戏区域偏移
        self.game_y_offset = 50

        # 蛇的参数 - 线段表示
        self.snake_line = []  # 蛇的线段点列表
        self.max_snake_length = 200  # 蛇的最大长度（像素）
        self.current_length = 0  # 当前蛇的长度
        self.segment_length = 5  # 每个线段段的长度
        self.snake_thickness = 15  # 蛇的厚度
        self.score = 0
        self.game_over = False
        self.game_started = False

        # 食物
        self.food_pos = None
        self.food_spawned = False
        self.food_radius = 15  # 食物半径

        # 手指追踪
        self.finger_pos = None  # 手指位置
        self.last_finger_pos = None
        self.finger_detected = False

        # 游戏速度
        self.game_speed = 60  # 游戏帧率
        self.last_update_time = time.time()

        # 初始位置标志
        self.initial_position_set = False

    def init_game_area(self, frame):
        """初始化游戏区域"""
        h, w = frame.shape[:2]
        self.game_width = w - 2 * self.game_x_offset
        self.game_height = h - 2 * self.game_y_offset

        print(f"游戏区域: {self.game_width}x{self.game_height}")

    def init_snake_at_position(self, start_x, start_y):
        """在指定位置初始化蛇（线段）"""
        print(f"在位置 ({start_x}, {start_y}) 初始化蛇")  # 修复：使用start_y而不是y

        # 初始化蛇的线段 - 创建一个小线段
        self.snake_line = []
        for i in range(20):  # 初始20个点，形成一个小线段
            point_x = start_x - i * 2  # 稍微向左偏移创建初始线段
            point_y = start_y
            self.snake_line.append((point_x, point_y))

        self.current_length = len(self.snake_line) * self.segment_length

        # 生成第一个食物
        self.spawn_food()

        # 标记游戏已开始
        self.game_started = True
        self.initial_position_set = True
        print(f"蛇已初始化，长度: {self.current_length}")

    def spawn_food(self):
        """生成食物"""
        max_attempts = 100
        attempts = 0

        while attempts < max_attempts:
            # 在游戏区域内随机生成食物位置
            x = random.randint(
                self.game_x_offset + self.food_radius,
                self.game_x_offset + self.game_width - self.food_radius
            )
            y = random.randint(
                self.game_y_offset + self.food_radius,
                self.game_y_offset + self.game_height - self.food_radius
            )

            # 确保食物不在蛇身上
            food_on_snake = False
            if self.snake_line:
                # 检查食物是否离蛇太近
                for point in self.snake_line:
                    distance = math.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
                    if distance < self.snake_thickness + self.food_radius:
                        food_on_snake = True
                        break

            if not food_on_snake:
                self.food_pos = (x, y)
                self.food_spawned = True
                print(f"食物生成在: ({x}, {y})")
                return

            attempts += 1

        # 如果找不到合适的位置，尝试其他位置
        print("使用备用食物生成方法")
        all_positions = []
        grid_size = self.food_radius * 2

        for x in range(self.game_x_offset + self.food_radius,
                       self.game_x_offset + self.game_width - self.food_radius,
                       grid_size):
            for y in range(self.game_y_offset + self.food_radius,
                           self.game_y_offset + self.game_height - self.food_radius,
                           grid_size):

                # 检查该位置是否在蛇身上
                position_occupied = False
                if self.snake_line:
                    for point in self.snake_line:
                        distance = math.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
                        if distance < self.snake_thickness + self.food_radius:
                            position_occupied = True
                            break

                if not position_occupied:
                    all_positions.append((x, y))

        if all_positions:
            self.food_pos = random.choice(all_positions)
            self.food_spawned = True
            print(f"备用食物生成在: {self.food_pos}")
        else:
            print("警告：没有可用位置放置食物，稍后重试")

    def update_finger_position(self, finger_x, finger_y):
        """更新手指位置"""
        # 确保手指位置在游戏区域内
        finger_x = max(self.game_x_offset + self.snake_thickness // 2,
                       min(finger_x, self.game_x_offset + self.game_width - self.snake_thickness // 2))
        finger_y = max(self.game_y_offset + self.snake_thickness // 2,
                       min(finger_y, self.game_y_offset + self.game_height - self.snake_thickness // 2))

        self.finger_pos = (finger_x, finger_y)
        self.finger_detected = True

        # 如果游戏没有开始，且手指位置已设置，初始化蛇
        if not self.game_started and not self.initial_position_set:
            # 等待手指稳定后再初始化
            if self.last_finger_pos:
                dx = finger_x - self.last_finger_pos[0]
                dy = finger_y - self.last_finger_pos[1]
                if abs(dx) < 5 and abs(dy) < 5:  # 手指基本稳定
                    self.init_snake_at_position(finger_x, finger_y)
            else:
                self.last_finger_pos = (finger_x, finger_y)

    def update_snake(self):
        """更新蛇的位置（蛇头跟随手指，蛇身作为线段跟随）"""
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # 如果没有检测到手指，使用上次的位置
        if not self.finger_detected and self.game_started:
            # 如果没有手指，蛇停在原地
            return False

        # 重置手指检测标志
        self.finger_detected = False

        # 获取当前手指位置
        if not self.finger_pos:
            return False

        finger_x, finger_y = self.finger_pos

        # 检查游戏边界
        if (finger_x < self.game_x_offset + self.snake_thickness // 2 or
                finger_x > self.game_x_offset + self.game_width - self.snake_thickness // 2 or
                finger_y < self.game_y_offset + self.snake_thickness // 2 or
                finger_y > self.game_y_offset + self.game_height - self.snake_thickness // 2):
            self.game_over = True
            return False

        # 更新蛇的线段
        if not self.snake_line:
            # 如果蛇线段为空，初始化
            self.snake_line.append((finger_x, finger_y))
        else:
            # 添加新的点（蛇头位置）
            self.snake_line.insert(0, (finger_x, finger_y))

            # 控制蛇的长度
            if len(self.snake_line) * self.segment_length > self.max_snake_length:
                # 移除尾部的点以保持最大长度
                while len(self.snake_line) * self.segment_length > self.max_snake_length:
                    self.snake_line.pop()

            # 更新当前长度
            self.current_length = len(self.snake_line) * self.segment_length

        # 检查是否吃到食物
        ate_food = False
        if self.food_spawned and self.food_pos and self.snake_line:
            food_x, food_y = self.food_pos
            head_x, head_y = self.snake_line[0]  # 蛇头位置

            # 计算蛇头到食物的距离
            distance = math.sqrt((head_x - food_x) ** 2 + (head_y - food_y) ** 2)

            # 如果距离小于阈值，认为吃到食物
            if distance < self.snake_thickness / 2 + self.food_radius:
                print(f"吃到食物！蛇头: ({head_x}, {head_y}), 食物: {self.food_pos}")
                self.score += 10

                # 增加蛇的最大长度
                self.max_snake_length += 50

                # 生成新食物
                self.spawn_food()
                ate_food = True

        # 检查是否撞到自己
        if self.check_self_collision():
            self.game_over = True
            return False

        return ate_food

    def check_self_collision(self):
        """检查蛇头是否撞到蛇身"""
        if len(self.snake_line) < 50:  # 避免初始阶段误判，需要足够长的蛇
            return False

        if not self.snake_line:
            return False

        head_x, head_y = self.snake_line[0]

        # 检查蛇头与蛇身其他部分的距离（跳过靠近头部的部分）
        for i in range(20, len(self.snake_line)):  # 跳过前20个点（靠近头部）
            point_x, point_y = self.snake_line[i]
            distance = math.sqrt((head_x - point_x) ** 2 + (head_y - point_y) ** 2)

            if distance < self.snake_thickness:  # 如果距离太近，认为碰撞
                return True

        return False

    def draw_game(self, frame):
        """绘制游戏元素"""
        # 绘制游戏区域边框
        cv2.rectangle(
            frame,
            (self.game_x_offset - 2, self.game_y_offset - 2),
            (self.game_x_offset + self.game_width + 2,
             self.game_y_offset + self.game_height + 2),
            (0, 255, 0), 2
        )

        # 绘制蛇（线段）
        if len(self.snake_line) > 1:
            # 绘制蛇身（线段）
            for i in range(len(self.snake_line) - 1):
                # 计算当前点的颜色（渐变）
                color_ratio = i / len(self.snake_line)
                green_value = int(100 + 155 * (1 - color_ratio))

                # 绘制线段
                cv2.line(
                    frame,
                    (int(self.snake_line[i][0]), int(self.snake_line[i][1])),
                    (int(self.snake_line[i + 1][0]), int(self.snake_line[i + 1][1])),
                    (0, green_value, 0),
                    self.snake_thickness
                )

            # 绘制蛇头（更亮更大）
            if self.snake_line:
                head_x, head_y = self.snake_line[0]

                # 绘制蛇头（圆形）
                cv2.circle(
                    frame,
                    (int(head_x), int(head_y)),
                    self.snake_thickness + 2,
                    (0, 255, 0),
                    -1
                )

                # 绘制蛇头眼睛
                eye_radius = self.snake_thickness // 4
                eye_offset = self.snake_thickness // 2

                # 根据移动方向绘制眼睛
                if len(self.snake_line) > 1:
                    # 计算移动方向
                    next_x, next_y = self.snake_line[1]
                    dx = next_x - head_x
                    dy = next_y - head_y

                    # 归一化方向向量
                    length = math.sqrt(dx * dx + dy * dy)
                    if length > 0:
                        dx = dx / length
                        dy = dy / length

                        # 计算眼睛位置（垂直于移动方向）
                        perp_dx = -dy
                        perp_dy = dx

                        # 绘制两只眼睛
                        cv2.circle(
                            frame,
                            (int(head_x + perp_dx * eye_offset), int(head_y + perp_dy * eye_offset)),
                            eye_radius,
                            (0, 0, 0),
                            -1
                        )
                        cv2.circle(
                            frame,
                            (int(head_x - perp_dx * eye_offset), int(head_y - perp_dy * eye_offset)),
                            eye_radius,
                            (0, 0, 0),
                            -1
                        )

        # 绘制手指位置指示器
        if self.finger_pos:
            fx, fy = self.finger_pos
            # 绘制目标位置
            cv2.circle(frame, (int(fx), int(fy)), 8, (255, 0, 255), -1)  # 紫色圆点
            cv2.circle(frame, (int(fx), int(fy)), 12, (255, 0, 255), 2)  # 紫色圆圈

        # 绘制食物
        if self.food_spawned and self.food_pos:
            fx, fy = self.food_pos

            # 绘制食物（苹果形状）
            cv2.circle(
                frame,
                (int(fx), int(fy)),
                self.food_radius,
                (0, 0, 255),
                -1
            )

            # 苹果茎
            cv2.rectangle(
                frame,
                (int(fx) - 1, int(fy - self.food_radius)),
                (int(fx) + 1, int(fy - self.food_radius - 5)),
                (139, 69, 19),
                -1
            )

            # 苹果叶子
            leaf_points = np.array([
                [int(fx), int(fy - self.food_radius)],
                [int(fx - 3), int(fy - self.food_radius - 3)],
                [int(fx - 1), int(fy - self.food_radius - 4)]
            ], np.int32)
            cv2.fillPoly(frame, [leaf_points], (34, 139, 34))

            # 食物光泽
            cv2.circle(
                frame,
                (int(fx - self.food_radius // 4), int(fy - self.food_radius // 4)),
                self.food_radius // 6,
                (255, 255, 255),
                -1
            )

        # 绘制分数和状态
        cv2.putText(
            frame,
            f"Score: {self.score}",
            (self.game_x_offset, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Length: {self.current_length}",
            (self.game_x_offset + 200, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Max Length: {self.max_snake_length}",
            (self.game_x_offset + 400, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        if self.game_over:
            cv2.putText(
                frame,
                "GAME OVER! Press 'R' to restart",
                (self.game_x_offset + self.game_width // 2 - 200,
                 self.game_y_offset + self.game_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )
            cv2.putText(
                frame,
                f"Final Score: {self.score}",
                (self.game_x_offset + self.game_width // 2 - 100,
                 self.game_y_offset + self.game_height // 2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2
            )
        elif not self.game_started:
            # 显示初始化提示
            cv2.putText(
                frame,
                "Show your hand and hold finger steady to create snake",
                (self.game_x_offset + self.game_width // 2 - 300,
                 self.game_y_offset + self.game_height // 2 - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2
            )
            cv2.putText(
                frame,
                "Then move finger to control the snake",
                (self.game_x_offset + self.game_width // 2 - 250,
                 self.game_y_offset + self.game_height // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2
            )
        else:
            # 显示控制提示
            cv2.putText(
                frame,
                "Move finger to control snake head",
                (self.game_x_offset + self.game_width // 2 - 150,
                 self.game_y_offset + self.game_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

    def reset_game(self):
        """重置游戏"""
        self.snake_line = []
        self.current_length = 0
        self.max_snake_length = 200
        self.score = 0
        self.game_over = False
        self.game_started = False
        self.finger_detected = False
        self.finger_pos = None
        self.food_spawned = False
        self.initial_position_set = False
        self.last_finger_pos = None
        print("游戏已重置")

    def run(self):
        """运行游戏主循环"""
        print("Starting Finger Controlled Snake Game...")
        print("Instructions:")
        print("1. Show your hand to the camera")
        print("2. Hold your finger steady to create a snake at finger position")
        print("3. Move finger to control the snake head")
        print("4. Eat red apples to grow and earn points")
        print("5. Avoid hitting the walls or yourself")
        print("6. Press 'R' to restart game")
        print("7. Press 'Q' or ESC to quit")

        # 初始化游戏区域（基于第一帧）
        ret, first_frame = self.cap.read()
        if ret:
            self.init_game_area(first_frame)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # 水平翻转画面
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # 转换为RGB用于Mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 为了减少MediaPipe警告，确保图像尺寸正确
            rgb_frame.flags.writeable = False

            results = self.hands.process(rgb_frame)

            # 恢复可写标志
            rgb_frame.flags.writeable = True

            # 检测手部
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 绘制手部关键点
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )

                    # 获取食指指尖坐标（第8号关键点）
                    index_finger_tip = hand_landmarks.landmark[8]
                    finger_x = int(index_finger_tip.x * w)
                    finger_y = int(index_finger_tip.y * h)

                    # 绘制手指点
                    cv2.circle(frame, (finger_x, finger_y), 12, (255, 0, 0), -1)
                    cv2.circle(frame, (finger_x, finger_y), 15, (255, 255, 255), 2)

                    # 更新手指位置
                    self.update_finger_position(finger_x, finger_y)

                    # 在初始化阶段显示手指位置
                    if not self.game_started:
                        cv2.putText(
                            frame,
                            f"Finger at: ({finger_x}, {finger_y})",
                            (finger_x + 20, finger_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )

            # 更新和绘制游戏
            if self.game_started and not self.game_over:
                ate_food = self.update_snake()
                if ate_food:
                    print(f"吃到食物！当前分数: {self.score}")

            # 绘制游戏元素
            self.draw_game(frame)

            # 显示帮助文本
            cv2.putText(
                frame,
                "Hold finger steady to create snake | R: Restart | Q: Quit",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )

            # 显示窗口
            cv2.imshow('Finger Controlled Snake Game - Line Segment', frame)

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r') or key == ord('R'):
                self.reset_game()
            elif key == ord('q') or key == ord('Q') or key == 27:  # ESC键
                break

        # 清理资源
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    game = FingerControlledSnakeGame()
    game.run()