# main.py

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import numpy as np
import heapq
from collections import defaultdict
import time
from queue import PriorityQueue
import random


# 基础配置和枚举类型定义
class CorridorType(Enum):
    WIDE = "wide"  # 宽走廊，可双向通行
    NARROW = "narrow"  # 窄走廊，单向通行


class AreaType(Enum):
    LOADING = "loading"  # 装载区
    UNLOADING = "unloading"  # 卸载区
    STORAGE = "storage"  # 存储区
    CORRIDOR = "corridor"  # 走廊
    INTERSECTION = "intersection"  # 交叉点


class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    EXECUTING = "executing"
    COMPLETED = "completed"


class RobotStatus(Enum):
    IDLE = "idle"
    MOVING = "moving"
    WORKING = "working"


class MapConfig:
    """地图配置"""
    SIZE = 100  # 地图大小
    NUM_ROBOTS = 10  # 机器人数量
    NUM_TASKS = 45  # 固定任务数量

    # 走廊配置
    WIDE_CORRIDOR_WIDTH = 3
    NARROW_CORRIDOR_WIDTH = 1

    # 区域配置
    AREA_CAPACITY = {
        AreaType.LOADING: 5,
        AreaType.UNLOADING: 5,
        AreaType.STORAGE: 10
    }

    # 时间配置
    TIME_STEP = 1
    MAX_SIMULATION_TIME = 1000

    # CBS配置
    MAX_CBS_ITERATIONS = 1000
    CBS_TIMEOUT = 30  # seconds


@dataclass
class Position:
    """位置类"""
    x: int
    y: int

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def distance_to(self, other: 'Position') -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def __add__(self, other):
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Position(self.x - other.x, self.y - other.y)


@dataclass
class Corridor:
    """走廊类"""
    id: str
    type: CorridorType
    start: Position
    end: Position
    width: int
    direction: Optional[Tuple[int, int]] = None  # 单向走廊的方向
    capacity: int = field(init=False)

    def __post_init__(self):
        self.capacity = 2 if self.type == CorridorType.WIDE else 1


@dataclass
class WorkArea:
    """工作区域类"""
    id: str
    type: AreaType
    position: Position
    capacity: int
    current_occupancy: int = 0
    task_points: Dict[str, Position] = field(default_factory=dict)


@dataclass
class Intersection:
    """交叉点类"""
    id: str
    position: Position
    connected_corridors: List[str]  # 相连走廊的ID列表
    capacity: int = 1  # 交叉点同时允许的机器人数量


@dataclass
class Task:
    """任务类"""
    id: int
    type: int  # 1: 装载任务, 2: 卸载任务
    start_point: str  # 任务起点ID
    end_point: str  # 任务终点ID
    priority: float = 0.0
    status: TaskStatus = TaskStatus.PENDING
    assigned_robot: Optional[int] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None

    def __hash__(self):
        return hash(self.id)


@dataclass
class Robot:
    """机器人类"""
    id: int
    position: Position
    status: RobotStatus = RobotStatus.IDLE
    current_task: Optional[Task] = None
    path: List[Position] = field(default_factory=list)
    path_index: int = 0

    def __hash__(self):
        return hash(self.id)


@dataclass
class CBSConstraint:
    """CBS约束"""
    robot_id: int
    position: Position
    time_step: int

    def __hash__(self):
        return hash((self.robot_id, self.position, self.time_step))


@dataclass
class CBSConflict:
    """CBS冲突"""
    time_step: int
    robot1_id: int
    robot2_id: int
    position: Position


@dataclass
class CBSNode:
    """CBS节点"""
    constraints: Set[CBSConstraint] = field(default_factory=set)
    paths: Dict[int, List[Position]] = field(default_factory=dict)
    cost: float = float('inf')

    def __lt__(self, other):
        return self.cost < other.cost


class WarehouseMap:
    """仓库地图系统"""

    def __init__(self):
        self.size = MapConfig.SIZE
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.corridors: Dict[str, Corridor] = {}
        self.areas: Dict[str, WorkArea] = {}
        self.intersections: Dict[str, Intersection] = {}
        self.task_points: Dict[str, Position] = {}

    def initialize(self):
        """初始化地图"""
        try:
            print("Initializing map components...")
            self._setup_corridors()
            print("Corridors setup completed")

            self._setup_work_areas()
            print("Work areas setup completed")

            self._setup_task_points()
            print("Task points setup completed")

            self._mark_intersections()
            print("Intersections marked")

            self._update_grid()
            print("Grid updated")

            # 验证地图初始化
            valid_positions = self._count_valid_positions()
            print(f"Valid positions in map: {valid_positions}")

            if valid_positions < MapConfig.NUM_ROBOTS:
                print(f"Warning: Not enough valid positions ({valid_positions}) "
                      f"for {MapConfig.NUM_ROBOTS} robots")
                return False

            return True

        except Exception as e:
            print(f"Error in map initialization: {e}")
            return False

    def _count_valid_positions(self) -> int:
        """计算有效位置数量"""
        return np.sum(self.grid == 1)  # 1表示可通行区域

    def _setup_corridors(self):
        """设置走廊"""
        # 主要水平走廊（宽）
        self.corridors["W1"] = Corridor(
            id="W1",
            type=CorridorType.WIDE,
            start=Position(20, 0),
            end=Position(20, self.size - 1),
            width=MapConfig.WIDE_CORRIDOR_WIDTH
        )
        self.corridors["W2"] = Corridor(
            id="W2",
            type=CorridorType.WIDE,
            start=Position(80, 0),
            end=Position(80, self.size - 1),
            width=MapConfig.WIDE_CORRIDOR_WIDTH
        )

        # 垂直连接走廊（窄）
        self.corridors["N1"] = Corridor(
            id="N1",
            type=CorridorType.NARROW,
            start=Position(0, 30),
            end=Position(self.size - 1, 30),
            width=MapConfig.NARROW_CORRIDOR_WIDTH,
            direction=(1, 0)
        )
        self.corridors["N2"] = Corridor(
            id="N2",
            type=CorridorType.NARROW,
            start=Position(0, 70),
            end=Position(self.size - 1, 70),
            width=MapConfig.NARROW_CORRIDOR_WIDTH,
            direction=(-1, 0)
        )

        print(f"Created {len(self.corridors)} corridors")

    def _setup_work_areas(self):
        """设置工作区域"""
        # 装载区
        self.areas["L1"] = WorkArea(
            id="L1",
            type=AreaType.LOADING,
            position=Position(15, 15),
            capacity=MapConfig.AREA_CAPACITY[AreaType.LOADING]
        )
        self.areas["L2"] = WorkArea(
            id="L2",
            type=AreaType.LOADING,
            position=Position(85, 85),
            capacity=MapConfig.AREA_CAPACITY[AreaType.LOADING]
        )

        # 卸载区
        self.areas["U1"] = WorkArea(
            id="U1",
            type=AreaType.UNLOADING,
            position=Position(15, 85),
            capacity=MapConfig.AREA_CAPACITY[AreaType.UNLOADING]
        )
        self.areas["U2"] = WorkArea(
            id="U2",
            type=AreaType.UNLOADING,
            position=Position(85, 15),
            capacity=MapConfig.AREA_CAPACITY[AreaType.UNLOADING]
        )

        print(f"Created {len(self.areas)} work areas")

    def _setup_task_points(self):
        """设置任务点"""
        task_point_count = 0
        for area in self.areas.values():
            # 为每个工作区域创建固定的任务点
            area_points = 0
            for i in range(area.capacity):
                point_id = f"{area.id}_P{i}"
                # 在工作区域周围创建任务点，确保不重叠
                offset_x = (i % 3) * 2
                offset_y = (i // 3) * 2

                pos = Position(
                    area.position.x + offset_x,
                    area.position.y + offset_y
                )

                # 确保位置在地图范围内
                if (0 <= pos.x < self.size and 0 <= pos.y < self.size):
                    self.task_points[point_id] = pos
                    area.task_points[point_id] = pos
                    area_points += 1
                    task_point_count += 1

            print(f"Created {area_points} task points for area {area.id}")

        print(f"Total task points created: {task_point_count}")

    def _mark_intersections(self):
        """标记交叉点"""
        intersection_count = 0
        for c1_id, c1 in self.corridors.items():
            for c2_id, c2 in self.corridors.items():
                if c1_id >= c2_id:
                    continue

                intersection = self._find_intersection(c1, c2)
                if intersection:
                    int_id = f"I_{c1_id}_{c2_id}"
                    self.intersections[int_id] = Intersection(
                        id=int_id,
                        position=intersection,
                        connected_corridors=[c1_id, c2_id]
                    )
                    intersection_count += 1

        print(f"Found {intersection_count} intersections")

    def _find_intersection(self, c1: Corridor, c2: Corridor) -> Optional[Position]:
        """查找两条走廊的交叉点"""
        # 水平和垂直走廊的交叉检测
        if ((c1.start.x == c1.end.x and c2.start.y == c2.end.y) or
                (c1.start.y == c1.end.y and c2.start.x == c2.end.x)):

            # 确定交叉点坐标
            if c1.start.x == c1.end.x:  # c1是垂直走廊
                x = c1.start.x
                y = c2.start.y
            else:  # c1是水平走廊
                x = c2.start.x
                y = c1.start.y

            # 检查交叉点是否在两条走廊的范围内
            if (min(c1.start.x, c1.end.x) <= x <= max(c1.start.x, c1.end.x) and
                    min(c1.start.y, c1.end.y) <= y <= max(c1.start.y, c1.end.y) and
                    min(c2.start.x, c2.end.x) <= x <= max(c2.start.x, c2.end.x) and
                    min(c2.start.y, c2.end.y) <= y <= max(c2.start.y, c2.end.y)):
                return Position(x, y)

        return None

    def _update_grid(self):
        """更新网格地图"""
        # 重置网格
        self.grid.fill(0)

        # 标记走廊
        for corridor in self.corridors.values():
            self._mark_corridor(corridor)

        # 标记工作区域
        for area in self.areas.values():
            self._mark_area(area)

        # 确保任务点是可通行的
        for point in self.task_points.values():
            if 0 <= point.x < self.size and 0 <= point.y < self.size:
                self.grid[point.x, point.y] = 1

        # 标记交叉点
        for intersection in self.intersections.values():
            pos = intersection.position
            if 0 <= pos.x < self.size and 0 <= pos.y < self.size:
                self.grid[pos.x, pos.y] = 1

        # 打印网格统计信息
        valid_cells = np.sum(self.grid == 1)
        print(f"Grid updated: {valid_cells} valid cells marked")

    def _mark_corridor(self, corridor: Corridor):
        """在网格上标记走廊"""
        if corridor.start.x == corridor.end.x:  # 垂直走廊
            x = corridor.start.x
            y_start = min(corridor.start.y, corridor.end.y)
            y_end = max(corridor.start.y, corridor.end.y)

            for y in range(y_start, y_end + 1):
                for w in range(-corridor.width // 2, corridor.width // 2 + 1):
                    if 0 <= x + w < self.size:
                        self.grid[x + w, y] = 1
        else:  # 水平走廊
            y = corridor.start.y
            x_start = min(corridor.start.x, corridor.end.x)
            x_end = max(corridor.start.x, corridor.end.x)

            for x in range(x_start, x_end + 1):
                for w in range(-corridor.width // 2, corridor.width // 2 + 1):
                    if 0 <= y + w < self.size:
                        self.grid[x, y + w] = 1

    def _mark_area(self, area: WorkArea):
        """在网格上标记工作区域"""
        x, y = area.position.x, area.position.y

        # 标记工作区域中心及周围区域
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < self.size and
                        0 <= new_y < self.size):
                    self.grid[new_x, new_y] = 1

    def is_valid_position(self, pos: Position) -> bool:
        """检查位置是否有效"""
        return (0 <= pos.x < self.size and
                0 <= pos.y < self.size and
                self.grid[pos.x, pos.y] == 1)

    def get_neighbors(self, pos: Position) -> List[Position]:
        """获取相邻的有效位置"""
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 四个方向

        for dx, dy in directions:
            new_pos = Position(pos.x + dx, pos.y + dy)
            if self.is_valid_position(new_pos):
                neighbors.append(new_pos)

        return neighbors

    def print_map_status(self):
        """打印地图状态"""
        print("\n=== Map Status ===")
        print(f"Corridors: {len(self.corridors)}")
        print(f"Work Areas: {len(self.areas)}")
        print(f"Task Points: {len(self.task_points)}")
        print(f"Intersections: {len(self.intersections)}")
        print(f"Valid Positions: {np.sum(self.grid == 1)}")


class TaskManager:
    """任务管理系统"""

    def __init__(self, warehouse_map: WarehouseMap):
        self.map = warehouse_map
        self.tasks: Dict[int, Task] = {}
        self.pending_tasks: Set[int] = set()
        self.assigned_tasks: Dict[int, int] = {}  # task_id -> robot_id
        self.completed_tasks: Set[int] = set()
        self.current_task_id = 1

    def generate_initial_tasks(self) -> bool:
        """生成初始任务集"""
        try:
            print("\nGenerating initial tasks...")

            # 获取所有装载点和卸载点
            loading_points = [point_id for point_id in self.map.task_points
                              if point_id.startswith("L")]
            unloading_points = [point_id for point_id in self.map.task_points
                                if point_id.startswith("U")]

            if not loading_points or not unloading_points:
                print("Error: No loading or unloading points available")
                return False

            print(f"Available loading points: {len(loading_points)}")
            print(f"Available unloading points: {len(unloading_points)}")

            # 生成固定数量的任务
            tasks_created = 0
            for i in range(MapConfig.NUM_TASKS):
                task_type = 1 if i < MapConfig.NUM_TASKS / 2 else 2

                if task_type == 1:
                    start_point = random.choice(loading_points)
                    end_point = random.choice(unloading_points)
                else:
                    start_point = random.choice(unloading_points)
                    end_point = random.choice(loading_points)

                # 验证任务点的有效性
                start_pos = self.map.task_points[start_point]
                end_pos = self.map.task_points[end_point]

                if not (self.map.is_valid_position(start_pos) and
                        self.map.is_valid_position(end_pos)):
                    print(f"Warning: Invalid task points detected for task {self.current_task_id}")
                    continue

                task = Task(
                    id=self.current_task_id,
                    type=task_type,
                    start_point=start_point,
                    end_point=end_point,
                    status=TaskStatus.PENDING,
                    start_time=time.time()
                )

                self.tasks[self.current_task_id] = task
                self.pending_tasks.add(self.current_task_id)
                self.current_task_id += 1
                tasks_created += 1

            self._update_task_priorities()
            print(f"Successfully created {tasks_created} tasks")
            return True

        except Exception as e:
            print(f"Error generating initial tasks: {e}")
            return False

    def _update_task_priorities(self):
        """更新任务优先级"""
        current_time = time.time()
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                # 基础优先级计算
                base_priority = 0.5

                # 根据任务类型调整优先级
                type_factor = 0.3 if task.type == 1 else 0.2

                # 根据等待时间调整优先级
                wait_time = current_time - task.start_time if task.start_time else 0
                wait_factor = min(0.3, wait_time / 100.0)

                # 计算最终优先级
                task.priority = base_priority + type_factor + wait_factor

    def assign_task(self, task_id: int, robot_id: int) -> bool:
        """分配任务给机器人"""
        if task_id not in self.tasks or task_id not in self.pending_tasks:
            print(f"Cannot assign task {task_id}: Task not available")
            return False

        task = self.tasks[task_id]
        task.status = TaskStatus.ASSIGNED
        task.assigned_robot = robot_id

        self.pending_tasks.remove(task_id)
        self.assigned_tasks[task_id] = robot_id
        print(f"Task {task_id} assigned to robot {robot_id}")
        return True

    def complete_task(self, task_id: int) -> bool:
        """完成任务"""
        if task_id not in self.tasks or task_id not in self.assigned_tasks:
            print(f"Cannot complete task {task_id}: Task not assigned")
            return False

        task = self.tasks[task_id]
        task.status = TaskStatus.COMPLETED
        task.completion_time = time.time()

        self.assigned_tasks.pop(task_id)
        self.completed_tasks.add(task_id)
        print(f"Task {task_id} completed")
        return True

    def get_task_status(self, task_id: int) -> Optional[TaskStatus]:
        """获取任务状态"""
        if task_id in self.tasks:
            return self.tasks[task_id].status
        return None

    def print_task_statistics(self):
        """打印任务统计信息"""
        print("\n=== Task Statistics ===")
        print(f"Total tasks: {len(self.tasks)}")
        print(f"Pending tasks: {len(self.pending_tasks)}")
        print(f"Assigned tasks: {len(self.assigned_tasks)}")
        print(f"Completed tasks: {len(self.completed_tasks)}")

        # 计算平均等待时间和执行时间
        current_time = time.time()
        wait_times = []
        execution_times = []

        for task in self.tasks.values():
            if task.start_time:
                if task.status == TaskStatus.PENDING:
                    wait_times.append(current_time - task.start_time)
                elif task.status == TaskStatus.COMPLETED and task.completion_time:
                    execution_times.append(task.completion_time - task.start_time)

        if wait_times:
            avg_wait_time = sum(wait_times) / len(wait_times)
            print(f"Average waiting time: {avg_wait_time:.2f} seconds")

        if execution_times:
            avg_exec_time = sum(execution_times) / len(execution_times)
            print(f"Average execution time: {avg_exec_time:.2f} seconds")


class CBSPlanner:
    """CBS路径规划器"""

    def __init__(self, warehouse_map: WarehouseMap, task_manager: TaskManager):
        self.map = warehouse_map
        self.task_manager = task_manager
        self.time_limit = MapConfig.CBS_TIMEOUT
        self.start_time = 0
        print("CBS Planner initialized")

    def plan_paths(self, robots: Dict[int, Robot],
                   assignments: Dict[int, int]) -> Dict[int, List[Position]]:
        """使用CBS算法规划路径"""
        print(f"\nPlanning paths for {len(assignments)} robots")
        self.start_time = time.time()

        # 创建根节点
        root = CBSNode()

        # 为每个机器人计算初始路径
        for robot_id, task_id in assignments.items():
            robot = robots[robot_id]
            task = self.task_manager.tasks[task_id]

            # 计算到任务起点的路径
            start_point = self.map.task_points[task.start_point]
            path = self._compute_individual_path(
                robot.position,
                start_point,
                set()
            )

            if path:
                root.paths[robot_id] = path
                print(f"Initial path found for robot {robot_id}: {len(path)} steps")
            else:
                print(f"Warning: No initial path found for robot {robot_id}")
                return {}

        # 计算根节点代价
        root.cost = self._compute_solution_cost(root.paths)

        # 创建优先级队列
        queue = PriorityQueue()
        queue.put(root)

        iteration = 0
        while not queue.empty() and iteration < MapConfig.MAX_CBS_ITERATIONS:
            # 检查时间限制
            if time.time() - self.start_time > self.time_limit:
                print("CBS: Time limit exceeded")
                return self._get_best_solution(queue)

            node = queue.get()

            # 检查冲突
            conflict = self._find_first_conflict(node.paths)
            if not conflict:
                print(f"Solution found after {iteration} iterations")
                return node.paths

            # 处理冲突，生成子节点
            children = self._generate_child_nodes(node, conflict)
            for child in children:
                queue.put(child)

            iteration += 1
            if iteration % 100 == 0:
                print(f"CBS iteration {iteration}")

        print(f"CBS: Max iterations ({MapConfig.MAX_CBS_ITERATIONS}) reached")
        return self._get_best_solution(queue)

    def _compute_individual_path(self, start: Position, goal: Position,
                                 constraints: Set[CBSConstraint]) -> List[Position]:
        """使用A*算法计算单个机器人的路径"""

        def heuristic(pos: Position) -> float:
            return pos.distance_to(goal)

        def get_neighbors(pos: Position, time_step: int) -> List[Position]:
            neighbors = self.map.get_neighbors(pos)
            valid_neighbors = []
            for next_pos in neighbors:
                # 检查约束
                if not any(c.position == next_pos and c.time_step == time_step
                           for c in constraints):
                    valid_neighbors.append(next_pos)
            return valid_neighbors

        open_set = {start}
        closed_set = set()
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start)}

        while open_set:
            current = min(open_set, key=lambda pos: f_score[pos])

            if current == goal:
                # 重建路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            open_set.remove(current)
            closed_set.add(current)

            for next_pos in get_neighbors(current, g_score[current]):
                if next_pos in closed_set:
                    continue

                tentative_g_score = g_score[current] + 1

                if next_pos not in open_set:
                    open_set.add(next_pos)
                elif tentative_g_score >= g_score[next_pos]:
                    continue

                came_from[next_pos] = current
                g_score[next_pos] = tentative_g_score
                f_score[next_pos] = g_score[next_pos] + heuristic(next_pos)

        return []

    def _find_first_conflict(self, paths: Dict[int, List[Position]]) -> Optional[CBSConflict]:
        """查找路径中的第一个冲突"""
        max_path_length = max(len(path) for path in paths.values())

        # 检查每个时间步
        for t in range(max_path_length):
            # 获取每个机器人在时间t的位置
            positions = {}
            for robot_id, path in paths.items():
                if t < len(path):
                    pos = path[t]
                    # 检查顶点冲突
                    if pos in positions:
                        return CBSConflict(t, robot_id, positions[pos], pos)
                    positions[pos] = robot_id

                    # 检查边冲突
                    if t > 0 and t < len(path):
                        prev_pos = path[t - 1]
                        for other_id, other_path in paths.items():
                            if (other_id != robot_id and t < len(other_path) and
                                    other_path[t - 1] == pos and other_path[t] == prev_pos):
                                return CBSConflict(t, robot_id, other_id, pos)

        return None

    def _generate_child_nodes(self, parent: CBSNode,
                              conflict: CBSConflict) -> List[CBSNode]:
        """生成子节点"""
        children = []

        # 为冲突中的每个机器人生成一个子节点
        for robot_id in [conflict.robot1_id, conflict.robot2_id]:
            child = CBSNode(
                constraints=parent.constraints.copy(),
                paths=parent.paths.copy()
            )

            # 添加新约束
            new_constraint = CBSConstraint(
                robot_id=robot_id,
                position=conflict.position,
                time_step=conflict.time_step
            )
            child.constraints.add(new_constraint)

            # 重新规划受影响机器人的路径
            new_path = self._compute_individual_path(
                parent.paths[robot_id][0],  # 起点
                parent.paths[robot_id][-1],  # 终点
                child.constraints
            )

            if new_path:
                child.paths[robot_id] = new_path
                child.cost = self._compute_solution_cost(child.paths)
                children.append(child)

        return children

    def _compute_solution_cost(self, paths: Dict[int, List[Position]]) -> float:
        """计算解决方案的总代价"""
        return sum(len(path) for path in paths.values())

    def _get_best_solution(self, queue: PriorityQueue) -> Dict[int, List[Position]]:
        """获取队列中的最佳解决方案"""
        best_solution = {}
        best_cost = float('inf')

        while not queue.empty():
            node = queue.get()
            if node.cost < best_cost:
                best_cost = node.cost
                best_solution = node.paths

        return best_solution


class RobotController:
    """机器人控制系统"""

    def __init__(self, warehouse_map: WarehouseMap):
        self.map = warehouse_map
        self.robots: Dict[int, Robot] = {}
        print("Robot Controller initialized")

    def initialize_robots(self):
        """初始化机器人"""
        print("\nInitializing robots...")
        # 在走廊入口处初始化机器人
        available_positions = self._get_initial_positions()

        if not available_positions:
            print("Error: No valid positions found for robots")
            return False

        print(f"Found {len(available_positions)} valid initial positions")

        robots_initialized = 0
        for i in range(MapConfig.NUM_ROBOTS):
            if i < len(available_positions):
                position = available_positions[i]
                robot = Robot(
                    id=i + 1,
                    position=position,
                    status=RobotStatus.IDLE
                )
                self.robots[robot.id] = robot
                robots_initialized += 1
                print(f"Robot {robot.id} initialized at position ({position.x}, {position.y})")
            else:
                print(f"Warning: Could not initialize robot {i + 1}")

        print(f"Successfully initialized {robots_initialized} robots")
        return robots_initialized > 0

    def _get_initial_positions(self) -> List[Position]:
        """获取机器人的初始位置"""
        positions = []

        # 首先尝试在走廊入口处放置机器人
        for corridor in self.map.corridors.values():
            if corridor.type == CorridorType.WIDE:
                # 在宽走廊的起点和终点附近寻找位置
                for offset in range(-1, 2):
                    # 起点附近
                    start_pos = Position(
                        corridor.start.x + offset,
                        corridor.start.y
                    )
                    if self.map.is_valid_position(start_pos):
                        positions.append(start_pos)

                    # 终点附近
                    end_pos = Position(
                        corridor.end.x + offset,
                        corridor.end.y
                    )
                    if self.map.is_valid_position(end_pos):
                        positions.append(end_pos)

        # 如果走廊入口位置不够，在地图上寻找其他有效位置
        if len(positions) < MapConfig.NUM_ROBOTS:
            print("Looking for additional valid positions...")
            for x in range(0, self.map.size, 5):
                for y in range(0, self.map.size, 5):
                    if self.map.grid[x, y] == 1:  # 如果是可通行区域
                        pos = Position(x, y)
                        if pos not in positions:
                            positions.append(pos)
                            if len(positions) >= MapConfig.NUM_ROBOTS:
                                break
                if len(positions) >= MapConfig.NUM_ROBOTS:
                    break

        print(f"Found {len(positions)} total valid positions")
        return positions

    def update_robot_positions(self, current_time: int):
        """更新机器人位置"""
        for robot in self.robots.values():
            if robot.status == RobotStatus.MOVING and robot.path:
                if robot.path_index < len(robot.path):
                    # 移动到路径中的下一个位置
                    next_pos = robot.path[robot.path_index]

                    # 检查移动是否有效
                    if self._is_move_valid(robot.position, next_pos):
                        print(f"Robot {robot.id} moving from ({robot.position.x}, {robot.position.y}) "
                              f"to ({next_pos.x}, {next_pos.y})")
                        robot.position = next_pos
                        robot.path_index += 1

                        # 检查是否到达目标
                        if robot.path_index >= len(robot.path):
                            if robot.current_task:
                                robot.status = RobotStatus.WORKING
                                print(f"Robot {robot.id} reached task location and started working")
                            else:
                                robot.status = RobotStatus.IDLE
                                robot.path = []
                                robot.path_index = 0
                                print(f"Robot {robot.id} completed movement and is now idle")
                    else:
                        print(f"Invalid move detected for robot {robot.id}, replanning needed")
                        robot.status = RobotStatus.IDLE
                        robot.path = []
                        robot.path_index = 0

    def _is_move_valid(self, current_pos: Position, next_pos: Position) -> bool:
        """检查移动是否有效"""
        # 检查位置是否在地图范围内
        if not (0 <= next_pos.x < self.map.size and 0 <= next_pos.y < self.map.size):
            return False

        # 检查是否是可行走区域
        if self.map.grid[next_pos.x, next_pos.y] == 0:
            return False

        # 检查是否与其他机器人发生碰撞
        for other_robot in self.robots.values():
            if other_robot.position == next_pos:
                return False

        # 检查对角线移动是否有效
        if (abs(next_pos.x - current_pos.x) == 1 and
                abs(next_pos.y - current_pos.y) == 1):
            # 检查两个相邻的格子是否都是可行走的
            if (self.map.grid[current_pos.x, next_pos.y] == 0 or
                    self.map.grid[next_pos.x, current_pos.y] == 0):
                return False

        return True

    def assign_path_to_robot(self, robot_id: int, path: List[Position]) -> bool:
        """为机器人分配路径"""
        if robot_id not in self.robots:
            print(f"Error: Robot {robot_id} not found")
            return False

        robot = self.robots[robot_id]
        robot.path = path
        robot.path_index = 0
        robot.status = RobotStatus.MOVING
        print(f"Path assigned to robot {robot_id}: {len(path)} steps")
        return True

    def get_idle_robots(self) -> List[Robot]:
        """获取空闲机器人"""
        idle_robots = [robot for robot in self.robots.values()
                       if robot.status == RobotStatus.IDLE]
        print(f"Found {len(idle_robots)} idle robots")
        return idle_robots

    def get_robot_status(self, robot_id: int) -> Optional[RobotStatus]:
        """获取机器人状态"""
        if robot_id in self.robots:
            return self.robots[robot_id].status
        return None

    def print_robot_status(self):
        """打印所有机器人的状态"""
        print("\n=== Robot Status ===")
        status_count = defaultdict(int)
        for robot in self.robots.values():
            status_count[robot.status] += 1
            print(f"Robot {robot.id}: {robot.status.value} at ({robot.position.x}, {robot.position.y})")

        print("\nStatus Summary:")
        for status, count in status_count.items():
            print(f"{status.value}: {count}")


class WarehouseSystem:
    """仓库系统主类"""

    def __init__(self):
        self.map = WarehouseMap()
        self.task_manager = TaskManager(self.map)
        self.robot_controller = RobotController(self.map)
        self.cbs_planner = None  # 将在初始化后创建
        self.current_time = 0
        self.running = False
        print("Warehouse System initialized")

    def initialize(self):
        """初始化系统"""
        print("\nInitializing warehouse system...")

        try:
            # 初始化地图
            if not self.map.initialize():
                print("Failed to initialize map")
                return False
            print("Map initialized successfully")

            # 生成初始任务
            if not self.task_manager.generate_initial_tasks():
                print("Failed to generate initial tasks")
                return False
            print("Initial tasks generated successfully")

            # 初始化机器人控制器
            if not self.robot_controller.initialize_robots():
                print("Failed to initialize robots")
                return False
            print("Robots initialized successfully")

            # 初始化CBS规划器
            self.cbs_planner = CBSPlanner(self.map, self.task_manager)
            print("CBS Planner initialized successfully")

            return True

        except Exception as e:
            print(f"Error during system initialization: {e}")
            return False

    def run(self):
        """运行系统"""
        print("\nStarting warehouse system simulation...")
        self.running = True

        try:
            while self.running and self.current_time < MapConfig.MAX_SIMULATION_TIME:
                self.current_time += MapConfig.TIME_STEP

                # 1. 更新任务优先级
                self.task_manager._update_task_priorities()

                # 2. 分配任务给空闲机器人
                self._assign_tasks()

                # 3. 更新机器人位置
                self.robot_controller.update_robot_positions(self.current_time)

                # 4. 检查任务完成情况
                self._check_task_completion()

                # 5. 定期输出状态
                if self.current_time % 10 == 0:
                    self._print_status()

                # 6. 检查终止条件
                if self._check_termination():
                    print("\nSimulation completed successfully")
                    break

        except Exception as e:
            print(f"\nError during simulation: {e}")
        finally:
            self._print_final_statistics()

    def _assign_tasks(self):
        """分配任务给空闲机器人"""
        idle_robots = self.robot_controller.get_idle_robots()
        if not idle_robots or not self.task_manager.pending_tasks:
            return

        print(f"\nAttempting to assign tasks to {len(idle_robots)} idle robots")

        # 获取待分配的任务
        assignments = {}
        for robot in idle_robots:
            if not self.task_manager.pending_tasks:
                break

            # 选择优先级最高的任务
            task_id = max(self.task_manager.pending_tasks,
                          key=lambda x: self.task_manager.tasks[x].priority)

            assignments[robot.id] = task_id
            print(f"Planning to assign task {task_id} to robot {robot.id}")

        if assignments:
            # 使用CBS规划路径
            paths = self.cbs_planner.plan_paths(self.robot_controller.robots, assignments)

            if paths:
                # 执行分配
                for robot_id, task_id in assignments.items():
                    if robot_id in paths:
                        if self.task_manager.assign_task(task_id, robot_id):
                            if self.robot_controller.assign_path_to_robot(robot_id, paths[robot_id]):
                                self.robot_controller.robots[robot_id].current_task = \
                                    self.task_manager.tasks[task_id]
                                print(f"Successfully assigned task {task_id} to robot {robot_id}")
                            else:
                                print(f"Failed to assign path to robot {robot_id}")
                        else:
                            print(f"Failed to assign task {task_id}")
            else:
                print("Failed to find valid paths for assignments")

    def _check_task_completion(self):
        """检查任务完成情况"""
        for robot in self.robot_controller.robots.values():
            if (robot.status == RobotStatus.WORKING and
                    robot.current_task and
                    robot.position == self.map.task_points[robot.current_task.end_point]):

                task_id = robot.current_task.id
                if self.task_manager.complete_task(task_id):
                    robot.status = RobotStatus.IDLE
                    robot.current_task = None
                    robot.path = []
                    robot.path_index = 0
                    print(f"Robot {robot.id} completed task {task_id}")

    def _check_termination(self) -> bool:
        """检查是否满足终止条件"""
        # 检查是否所有任务都已完成
        all_tasks_completed = (len(self.task_manager.completed_tasks) ==
                               MapConfig.NUM_TASKS)

        # 检查是否所有机器人都空闲
        all_robots_idle = all(robot.status == RobotStatus.IDLE
                              for robot in self.robot_controller.robots.values())

        if all_tasks_completed and all_robots_idle:
            print("\nAll tasks completed and all robots idle")
            return True
        return False

    def _print_status(self):
        """打印当前状态"""
        print(f"\n=== Time: {self.current_time} ===")
        print("Task Status:")
        print(f"Pending: {len(self.task_manager.pending_tasks)}")
        print(f"Assigned: {len(self.task_manager.assigned_tasks)}")
        print(f"Completed: {len(self.task_manager.completed_tasks)}")

        self.robot_controller.print_robot_status()

    def _print_final_statistics(self):
        """打印最终统计信息"""
        print("\n====== Final Statistics ======")
        print(f"Simulation time: {self.current_time}")

        # 任务统计
        self.task_manager.print_task_statistics()

        # 机器人统计
        self.robot_controller.print_robot_status()

        # 计算系统效率
        total_tasks = len(self.task_manager.completed_tasks)
        if self.current_time > 0:
            tasks_per_time = total_tasks / self.current_time
            print(f"\nSystem Efficiency:")
            print(f"Tasks completed per time unit: {tasks_per_time:.2f}")


def main():
    """主函数"""
    try:
        # 创建并初始化仓库系统
        print("Creating warehouse system...")
        system = WarehouseSystem()

        # 初始化系统
        if not system.initialize():
            print("Failed to initialize system")
            return

        # 运行系统
        system.run()

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 打印最终统计信息
        if 'system' in locals():
            system._print_final_statistics()


if __name__ == "__main__":
    main()