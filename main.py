import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import random
from collections import defaultdict
from datetime import datetime
import heapq
import time


class SimConfig:
    MAP_SIZE = 100
    NUM_ROBOTS = 10
    NUM_INITIAL_TASKS = 45  # 固定45个任务
    MAX_SIMULATION_TIME = 1000
    TIME_UNIT = 1
    MAX_REPLAN_ATTEMPTS = 3
    PRIORITY_UPDATE_INTERVAL = 5
    CHAIN_OPTIMIZATION_THRESHOLD = 15


@dataclass
class TaskRecord:
    """单个任务记录"""
    task_id: int
    task_type: int
    start_time: int
    completion_time: int
    waiting_time: float
    execution_time: float
    start_pos: Tuple[int, int]
    end_pos: Tuple[int, int]


class TaskSequenceTracker:
    """任务序列跟踪器"""

    def __init__(self):
        self.sequences: Dict[int, List[TaskRecord]] = {}  # robot_id -> task sequence
        self.completed_tasks: Dict[int, bool] = {}  # task_id -> completion status

    def record_task(self, robot_id: int, task_record: TaskRecord) -> None:
        """记录任务完成"""
        if robot_id not in self.sequences:
            self.sequences[robot_id] = []
        self.sequences[robot_id].append(task_record)
        self.completed_tasks[task_record.task_id] = True

    def get_sequence(self, robot_id: int) -> List[TaskRecord]:
        """获取机器人的任务序列"""
        return self.sequences.get(robot_id, [])

    def check_completion(self, total_tasks: List[int]) -> Tuple[bool, List[int]]:
        """检查任务完成情况"""
        uncompleted = [
            task_id for task_id in total_tasks
            if task_id not in self.completed_tasks
        ]
        return len(uncompleted) == 0, uncompleted

    def check_sequence_completeness(self) -> bool:
        """检查所有任务序列的完整性"""
        all_task_ids = set()
        for sequence in self.sequences.values():
            for record in sequence:
                all_task_ids.add(record.task_id)

        return len(all_task_ids) == SimConfig.NUM_INITIAL_TASKS

    def check_task_completion(self, robots: List['Robot']) -> Tuple[bool, List[int]]:
        """检查任务完成情况
        Args:
            robots: 机器人列表
        Returns:
            is_complete: 是否所有任务都已完成
            uncompleted: 未完成的任务ID列表
        """
        all_task_ids = set(range(1, SimConfig.NUM_INITIAL_TASKS + 1))
        completed_ids = set()

        # 收集所有已完成的任务ID
        for robot in robots:
            completed_ids.update(robot.completed_tasks)

        # 计算未完成的任务
        uncompleted = sorted(list(all_task_ids - completed_ids))
        return len(uncompleted) == 0, uncompleted


# Data structures
@dataclass
class Position:
    x: int
    y: int

    def __iter__(self):
        yield self.x
        yield self.y

    def __add__(self, other):
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Position(self.x - other.x, self.y - other.y)


@dataclass
class TaskSpecificParams:
    # For type 1 tasks (material transport)
    unload_time: float = 0.0
    full_rate: float = 0.0
    # For type 2 tasks (empty barrel transport)
    empty_rate: float = 0.0
    load_time: float = 0.0


@dataclass
class Task:
    id: int
    type: int  # 1: material transport, 2: empty barrel transport
    state: str  # new, open, assigned, completed
    start: Position
    end: Position
    open_time: int
    deadline: int
    priority: float = 0.0
    assigned_to: int = 0
    assign_time: int = 0
    start_time: int = 0
    complete_time: int = 0
    waiting_time: float = 0.0
    execution_time: float = 0.0
    total_time: float = 0.0
    specific: TaskSpecificParams = field(default_factory=TaskSpecificParams)
    next_task_id: int = 0


@dataclass
class Robot:
    id: int
    position: Position
    target: Position
    path: List[Position]
    state: str  # idle, moving, working
    current_task_id: int
    path_index: int
    last_update_time: int
    completed_tasks: List[int]  # 存储已完成的任务ID
    start_position: Position

    def get_task_statistics(self, tasks: List[Task]) -> Tuple[int, int, int, List[int]]:
        """获取任务统计信息
        Returns:
            total_completed: 完成总数
            type1_completed: 类型1完成数
            type2_completed: 类型2完成数
            completed_ids: 完成的任务ID列表
        """
        completed_ids = self.completed_tasks
        type1_count = sum(1 for task_id in completed_ids if tasks[task_id - 1].type == 1)
        type2_count = sum(1 for task_id in completed_ids if tasks[task_id - 1].type == 2)
        return len(completed_ids), type1_count, type2_count, sorted(completed_ids)


class TaskQueues:
    def __init__(self):
        self.all = []
        self.new = []
        self.open = []
        self.assigned = []
        self.completed = []


class System:
    def __init__(self):
        self.task_tracker = TaskSequenceTracker()
        self.map_matrix = np.zeros((SimConfig.MAP_SIZE, SimConfig.MAP_SIZE))
        self.robots: List[Robot] = []
        self.tasks: List[Task] = []
        self.task_queues = TaskQueues()
        self.global_time = 0

        # Additional tracking variables
        self.task_assignments = []
        self.task_completions = []
        self.conflicts = []
        self.system_performance = []

    def initialize(self):
        """Initialize the system with map, robots and initial tasks"""
        self.create_semantic_map()
        self.initialize_robots()
        self.generate_initial_tasks()
        self.calculate_dynamic_priorities()

        print(f"System initialized:")
        print(f"Map size: {SimConfig.MAP_SIZE}x{SimConfig.MAP_SIZE}")
        print(f"Number of robots: {SimConfig.NUM_ROBOTS}")
        print(f"Initial tasks: {SimConfig.NUM_INITIAL_TASKS}")

    def create_semantic_map(self):
        """Create semantic map with obstacles and corridors"""
        # Calculate number of obstacles (10% density)
        num_obstacles = round(SimConfig.MAP_SIZE * SimConfig.MAP_SIZE * 0.10)
        print(f"Generating {num_obstacles} obstacle areas...")

        obstacles_created = 0
        while obstacles_created < num_obstacles:
            x = random.randint(5, SimConfig.MAP_SIZE - 5)
            y = random.randint(5, SimConfig.MAP_SIZE - 5)
            width = random.randint(1, 3)
            height = random.randint(1, 3)

            # Check if area is available
            if (x + width <= SimConfig.MAP_SIZE and
                    y + height <= SimConfig.MAP_SIZE and
                    np.sum(self.map_matrix[x:min(x + width, SimConfig.MAP_SIZE),
                           y:min(y + height, SimConfig.MAP_SIZE)]) == 0):
                # Mark obstacle (4 represents obstacle)
                self.map_matrix[x:min(x + width, SimConfig.MAP_SIZE),
                y:min(y + height, SimConfig.MAP_SIZE)] = 4
                obstacles_created += 1

        print(f"Successfully generated {obstacles_created} obstacle areas")

        # Add corridors and partitions
        num_corridors = 4
        for _ in range(num_corridors):
            # Horizontal corridor
            y = random.randint(10, SimConfig.MAP_SIZE - 10)
            self.map_matrix[5:SimConfig.MAP_SIZE - 5, y:y + 2] = 0

            # Vertical corridor
            x = random.randint(10, SimConfig.MAP_SIZE - 10)
            self.map_matrix[x:x + 2, 5:SimConfig.MAP_SIZE - 5] = 0

        print(f"Added {num_corridors} corridors and {num_corridors} partitions")

    def initialize_robots(self):
        """Initialize robots and place them on the map"""
        print(f"Starting to place {SimConfig.NUM_ROBOTS} robots...")

        for i in range(SimConfig.NUM_ROBOTS):
            # Find valid position
            while True:
                x = random.randint(5, SimConfig.MAP_SIZE - 5)
                y = random.randint(5, SimConfig.MAP_SIZE - 5)
                if self.map_matrix[x, y] == 0:  # If position is empty
                    break

            # Update map (1 represents robot)
            self.map_matrix[x, y] = 1

            # Create robot
            position = Position(x, y)
            robot = Robot(
                id=i + 1,
                position=position,
                target=position,
                path=[],
                state='idle',
                current_task_id=0,
                path_index=1,
                last_update_time=0,
                completed_tasks=[],
                start_position=position
            )

            self.robots.append(robot)

        print(f"Successfully placed {SimConfig.NUM_ROBOTS} robots")

    def generate_initial_tasks(self):
        """Generate initial set of tasks"""
        print(f"Starting to generate {SimConfig.NUM_INITIAL_TASKS} initial tasks...")

        # Define work areas
        material_load_areas = [
            (10, 10, 20, 20),
            (80, 80, 90, 90)
        ]
        material_unload_areas = [
            (80, 10, 90, 20),
            (10, 80, 20, 90)
        ]
        empty_load_areas = [
            (30, 30, 40, 40),
            (60, 60, 70, 70)
        ]
        empty_unload_areas = [
            (60, 30, 70, 40),
            (30, 60, 40, 70)
        ]

        # 固定分配任务类型
        num_type1 = SimConfig.NUM_INITIAL_TASKS // 2  # 材料运输任务
        num_type2 = SimConfig.NUM_INITIAL_TASKS - num_type1  # 空桶运输任务

        for i in range(SimConfig.NUM_INITIAL_TASKS):
            task_type = 1 if i < num_type1 else 2

            if task_type == 1:
                start_area = random.choice(material_load_areas)
                end_area = random.choice(material_unload_areas)
                specific_params = TaskSpecificParams(
                    unload_time=50 + random.random() * 100,
                    full_rate=0.1 + random.random() * 0.4
                )
            else:
                start_area = random.choice(empty_load_areas)
                end_area = random.choice(empty_unload_areas)
                specific_params = TaskSpecificParams(
                    empty_rate=0.1 + random.random() * 0.4,
                    load_time=50 + random.random() * 100
                )

            # Find valid start position
            while True:
                start_x = random.randint(start_area[0], start_area[2])
                start_y = random.randint(start_area[1], start_area[3])
                if self.map_matrix[start_x, start_y] == 0:
                    break

            # Find valid end position
            while True:
                end_x = random.randint(end_area[0], end_area[2])
                end_y = random.randint(end_area[1], end_area[3])
                if self.map_matrix[end_x, end_y] == 0:
                    break

            # Create task
            task = Task(
                id=i + 1,
                type=task_type,
                state='open',  # 所有任务初始状态都设为open
                start=Position(start_x, start_y),
                end=Position(end_x, end_y),
                open_time=0,  # 所有任务的开始时间都设为0
                deadline=random.randint(200, 500),
                specific=specific_params
            )

            self.tasks.append(task)
            self.task_queues.all.append(task.id)
            self.task_queues.open.append(task.id)

            # Mark on map
            self.map_matrix[start_x, start_y] = 2
            self.map_matrix[end_x, end_y] = 3

        print(f"Successfully generated {SimConfig.NUM_INITIAL_TASKS} initial tasks "
              f"(Type 1: {num_type1}, Type 2: {num_type2})")

    def calculate_system_load(self) -> float:
        """Calculate current system load"""
        queue_length = len(self.task_queues.open)
        completed_length = len(self.task_queues.completed)
        queue_load = queue_length / max(30, queue_length + completed_length)

        busy_robots = sum(1 for robot in self.robots if robot.state != 'idle')
        robot_load = busy_robots / SimConfig.NUM_ROBOTS

        return 0.7 * queue_load + 0.3 * robot_load

    def manhattan_distance(self, pos1: Position, pos2: Position) -> int:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)

    def calculate_dynamic_priorities(self):
        """Calculate dynamic priorities for all open tasks"""
        if not self.task_queues.open:
            return

        system_load = self.calculate_system_load()

        # Separate tasks by type
        type1_tasks = []
        type2_tasks = []
        for task_id in self.task_queues.open:
            task = self.tasks[task_id - 1]
            if task.type == 1:
                type1_tasks.append(task_id)
            else:
                type2_tasks.append(task_id)

        # Process type 1 tasks (material transport)
        if type1_tasks:
            self._process_type1_tasks(type1_tasks, system_load)

        # Process type 2 tasks (empty barrel transport)
        if type2_tasks:
            self._process_type2_tasks(type2_tasks, system_load)

        # Sort open tasks by priority
        self.task_queues.open.sort(
            key=lambda x: self.tasks[x - 1].priority,
            reverse=True
        )

    def _process_type1_tasks(self, task_ids: List[int], system_load: float):
        """Process type 1 tasks for priority calculation"""
        tasks_data = []
        for task_id in task_ids:
            task = self.tasks[task_id - 1]
            tasks_data.append({
                'id': task_id,
                'unload_time': task.specific.unload_time,
                'full_rate': task.specific.full_rate,
                'open_time': self.global_time - task.open_time
            })

        # Calculate factors
        for data in tasks_data:
            waiting_factor = min(1.0, data['open_time'] / 100.0)
            urgency_factor = max(0, 1.0 - data['unload_time'] / 200.0)

            # Adjust weights based on system load
            if system_load > 0.7:
                w1, w2, w3 = 0.3, 0.4, 0.3
            elif system_load > 0.4:
                w1, w2, w3 = 0.5, 0.3, 0.2
            else:
                w1, w2, w3 = 0.7, 0.2, 0.1

            # Calculate final priority
            basic_priority = (
                    0.35 * (1 - data['unload_time'] / 200.0) +
                    0.15 * (1 - data['full_rate']) +
                    0.5 * waiting_factor
            )

            final_priority = (
                    w1 * basic_priority +
                    w2 * waiting_factor +
                    w3 * urgency_factor
            )

            # Update task priority
            self.tasks[data['id'] - 1].priority = round(final_priority * 100) / 100

    def _process_type2_tasks(self, task_ids: List[int], system_load: float):
        """Process type 2 tasks for priority calculation"""
        tasks_data = []
        for task_id in task_ids:
            task = self.tasks[task_id - 1]
            tasks_data.append({
                'id': task_id,
                'empty_rate': task.specific.empty_rate,
                'load_time': task.specific.load_time,
                'open_time': self.global_time - task.open_time
            })

        # Calculate factors
        for data in tasks_data:
            waiting_factor = min(1.0, data['open_time'] / 100.0)
            urgency_factor = max(0, 1.0 - data['load_time'] / 200.0)

            # Adjust weights based on system load
            if system_load > 0.7:
                w1, w2, w3 = 0.3, 0.4, 0.3
            elif system_load > 0.4:
                w1, w2, w3 = 0.5, 0.3, 0.2
            else:
                w1, w2, w3 = 0.7, 0.2, 0.1

            # Calculate final priority
            basic_priority = (
                    0.15 * (1 - data['empty_rate']) +
                    0.35 * (1 - data['load_time'] / 200.0) +
                    0.5 * waiting_factor
            )

            final_priority = (
                    w1 * basic_priority +
                    w2 * waiting_factor +
                    w3 * urgency_factor
            )

            # Update task priority
            self.tasks[data['id'] - 1].priority = round(final_priority * 100) / 100

    def a_star_planner(self, start: Position, goal: Position,
                       temp_obstacles: List[Tuple[int, int]] = None) -> Tuple[List[Position], int]:
        """A* path planning algorithm"""
        if start.x == goal.x and start.y == goal.y:
            return [start], 0

        # Check start and goal validity
        if not self._is_valid_position(start) or not self._is_valid_position(goal):
            return [], 0

        # Initialize data structures
        closed_set = set()
        open_set = {(start.x, start.y)}
        came_from = {}

        g_score = defaultdict(lambda: float('inf'))
        f_score = defaultdict(lambda: float('inf'))

        g_score[(start.x, start.y)] = 0
        f_score[(start.x, start.y)] = self.manhattan_distance(start, goal)

        # Define movement directions (8 directions)
        directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),  # Up, Right, Down, Left
            (1, 1), (1, -1), (-1, -1), (-1, 1)  # Diagonals
        ]

        while open_set:
            # Find node with lowest f_score
            current = min(open_set, key=lambda pos: f_score[pos])
            current_pos = Position(current[0], current[1])

            if current[0] == goal.x and current[1] == goal.y:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(Position(current[0], current[1]))
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path, len(path)

            open_set.remove(current)
            closed_set.add(current)

            # Check all neighbors
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)

                if (not (0 <= neighbor[0] < SimConfig.MAP_SIZE and
                         0 <= neighbor[1] < SimConfig.MAP_SIZE)):
                    continue

                if neighbor in closed_set:
                    continue

                if self.map_matrix[neighbor[0], neighbor[1]] == 4:
                    continue

                # Check diagonal movement
                if abs(dx) == 1 and abs(dy) == 1:
                    if (self.map_matrix[current[0], current[1] + dy] == 4 or
                            self.map_matrix[current[0] + dx, current[1]] == 4):
                        continue

                # Calculate tentative g_score
                if abs(dx) == 1 and abs(dy) == 1:
                    tentative_g_score = g_score[current] + 1.414  # Diagonal cost
                else:
                    tentative_g_score = g_score[current] + 1.0  # Straight cost

                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score[neighbor]:
                    continue

                # This path is the best until now. Record it!
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = (g_score[neighbor] +
                                     self.manhattan_distance(Position(neighbor[0], neighbor[1]), goal))

        return [], 0

    def _is_valid_position(self, pos: Position, check_obstacle: bool = False) -> bool:
        """Check if position is valid"""
        if not (0 <= pos.x < SimConfig.MAP_SIZE and 0 <= pos.y < SimConfig.MAP_SIZE):
            return False

        if check_obstacle and self.map_matrix[pos.x, pos.y] == 4:
            return False

        return True

    def assign_tasks(self):
        """Assign tasks to idle robots"""
        # Find idle robots
        idle_robots = [robot for robot in self.robots if robot.state == 'idle']

        if not idle_robots or not self.task_queues.open:
            return

        # Decide which assignment algorithm to use
        use_hungarian = (self.global_time % 50 == 0 and
                         len(idle_robots) > 1 and
                         len(self.task_queues.open) > 1)

        if use_hungarian:
            self._hungarian_assignment(idle_robots)
        elif len(idle_robots) == 1:
            # Single robot case: assign highest priority task
            self._assign_single_task(idle_robots[0].id, self.task_queues.open[0])
        else:
            # Multiple robots: use greedy algorithm
            self._greedy_assignment(idle_robots)

    def _hungarian_assignment(self, idle_robots: List[Robot]):
        """Hungarian algorithm for task assignment"""
        num_robots = len(idle_robots)
        # Copy the task queue to avoid dynamic changes
        task_queue = self.task_queues.open[:]
        num_tasks = min(len(task_queue), num_robots * 3)

        if num_robots == 0 or num_tasks == 0:
            return

        # Create cost matrix
        cost_matrix = np.zeros((num_robots, num_tasks))

        for i, robot in enumerate(idle_robots):
            robot_pos = robot.position
            for j, task_id in enumerate(task_queue[:num_tasks]):
                task = self.tasks[task_id - 1]

                # Calculate distance cost
                distance_cost = self.manhattan_distance(robot_pos, task.start)

                # Calculate waiting penalty
                waiting_penalty = -5 * min(1.0, (self.global_time - task.open_time) / 50.0)

                # Calculate urgency bonus
                if task.type == 1:
                    urgency_bonus = -5 * max(0, 1.0 - task.specific.unload_time / 100.0)
                else:
                    urgency_bonus = -5 * max(0, 1.0 - task.specific.load_time / 100.0)

                # Calculate priority bonus
                priority_bonus = -10 * task.priority

                # Total cost
                cost_matrix[i, j] = distance_cost + waiting_penalty + urgency_bonus + priority_bonus

        # Apply Hungarian algorithm
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Execute assignments
        for i, j in zip(row_ind, col_ind):
            if j < len(task_queue):
                robot_id = idle_robots[i].id
                task_id = task_queue[j]
                self._assign_single_task(robot_id, task_id)
            else:
                print(f"Warning: Invalid task index {j} (task queue length: {len(task_queue)})")

    def _greedy_assignment(self, idle_robots: List[Robot], dry_run: bool = False):
        """Greedy algorithm for task assignment"""
        assignments = []
        num_tasks = min(len(self.task_queues.open), len(idle_robots) * 3)

        if num_tasks == 0:
            return assignments

        candidate_tasks = self.task_queues.open[:num_tasks]
        assigned_tasks = set()

        for robot in idle_robots:
            robot_pos = robot.position
            best_task_id = 0
            best_cost = float('inf')

            # Find best task for current robot
            for task_id in candidate_tasks:
                if task_id in assigned_tasks:
                    continue

                task = self.tasks[task_id - 1]

                # Calculate costs similar to Hungarian algorithm
                distance_cost = self.manhattan_distance(robot_pos, task.start)
                waiting_penalty = -5 * min(1.0, (self.global_time - task.open_time) / 50.0)

                if task.type == 1:
                    urgency_bonus = -5 * max(0, 1.0 - task.specific.unload_time / 100.0)
                else:
                    urgency_bonus = -5 * max(0, 1.0 - task.specific.load_time / 100.0)

                priority_bonus = -10 * task.priority
                total_cost = distance_cost + waiting_penalty + urgency_bonus + priority_bonus

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_task_id = task_id

            if best_task_id:
                assignments.append((robot.id, best_task_id))
                assigned_tasks.add(best_task_id)

        if dry_run:
            return assignments

        # Execute assignments
        for robot_id, task_id in assignments:
            self._assign_single_task(robot_id, task_id)

    def _assign_single_task(self, robot_id: int, task_id: int) -> bool:
        """Assign a specific task to a specific robot"""
        # Find robot and task
        robot = next(r for r in self.robots if r.id == robot_id)
        task = self.tasks[task_id - 1]

        # Check prerequisites
        if robot.state != 'idle' or task.state != 'open':
            return False

        # Calculate path to task start
        path, path_length = self.a_star_planner(robot.position, task.start)

        if not path:
            return False

        # Update robot status
        robot.state = 'moving'
        robot.target = task.start
        robot.path = path
        robot.path_index = 0
        robot.current_task_id = task_id
        robot.last_update_time = self.global_time

        # Update task status
        task.state = 'assigned'
        task.assigned_to = robot_id
        task.assign_time = self.global_time
        task.waiting_time = self.global_time - task.open_time

        # Update task queues
        self.task_queues.open.remove(task_id)
        self.task_queues.assigned.append(task_id)

        # Record assignment
        assignment_record = {
            'time': self.global_time,
            'task_id': task_id,
            'robot_id': robot_id,
            'priority': task.priority,
            'waiting_time': task.waiting_time,
            'path_length': path_length
        }
        self.task_assignments.append(assignment_record)

        print(f"Time {self.global_time}: Robot {robot_id} assigned to task {task_id} "
              f"[Priority: {task.priority:.2f}, Waiting: {task.waiting_time:.1f}, "
              f"Path length: {path_length}]")

        return True

    def update_robot_status(self):
        """Update status of all robots"""
        for robot in self.robots:
            elapsed_time = self.global_time - robot.last_update_time

            if robot.state in ['moving', 'working']:
                if robot.current_task_id == 0:
                    robot.state = 'idle'
                    robot.path = []
                    robot.last_update_time = self.global_time
                    continue

                task = self.tasks[robot.current_task_id - 1]

                # Handle robot movement
                if robot.path and robot.path_index < len(robot.path):
                    steps_to_move = min(elapsed_time, len(robot.path) - robot.path_index)

                    if steps_to_move > 0:
                        new_index = min(robot.path_index + steps_to_move, len(robot.path) - 1)
                        new_position = robot.path[new_index]

                        # Update map
                        self.map_matrix[robot.position.x, robot.position.y] = 0
                        self.map_matrix[new_position.x, new_position.y] = 1

                        robot.position = new_position
                        robot.path_index = new_index + 1

                    # Check if reached destination
                    if robot.path_index >= len(robot.path):
                        if robot.state == 'moving':
                            self._start_task_execution(robot)
                        elif robot.state == 'working':
                            self._complete_task(robot)

            robot.last_update_time = self.global_time

    def _start_task_execution(self, robot: Robot):
        """Start task execution when robot reaches task start point"""
        task = self.tasks[robot.current_task_id - 1]

        # Update robot status
        robot.state = 'working'
        robot.target = task.end

        # Plan path to task end point
        path, _ = self.a_star_planner(robot.position, task.end)
        robot.path = path
        robot.path_index = 0

        # Record task start time
        task.start_time = self.global_time

        print(f"Time {self.global_time}: Robot {robot.id} started executing task {task.id}")

    def _complete_task(self, robot: Robot):
        """任务完成处理"""
        task = self.tasks[robot.current_task_id - 1]

        # 添加到机器人的已完成任务列表
        robot.completed_tasks.append(task.id)

        # 更新任务状态
        task.state = 'completed'
        task.completion_time = self.global_time
        task.execution_time = self.global_time - task.start_time

        # 更新队列
        if task.id in self.task_queues.assigned:
            self.task_queues.assigned.remove(task.id)
        self.task_queues.completed.append(task.id)

        # 重置机器人状态
        robot.state = 'idle'
        robot.current_task_id = 0
        robot.path = []
        robot.path_index = 0

        # 创建任务记录并记录到追踪器
        task_record = TaskRecord(
            task_id=task.id,
            task_type=task.type,
            start_time=task.start_time,
            completion_time=self.global_time,
            waiting_time=task.waiting_time,
            execution_time=task.execution_time,
            start_pos=(task.start.x, task.start.y),
            end_pos=(task.end.x, task.end.y)
        )

        self.task_tracker.record_task(robot.id, task_record)

    def generate_dynamic_tasks(self):
        """Dynamically generate new tasks based on system load"""
        system_load = self.calculate_system_load()
        generation_probability = max(0.3, 1.0 - system_load)

        if random.random() < generation_probability:
            num_new_tasks = random.randint(0, 3)

            if num_new_tasks > 0:
                new_task_ids = []
                for _ in range(num_new_tasks):
                    task_id = len(self.tasks) + 1
                    task_type = random.randint(1, 2)

                    # Set areas based on task type
                    if task_type == 1:
                        if random.random() < 0.5:
                            start_area = (10, 10, 20, 20)
                        else:
                            start_area = (80, 80, 90, 90)

                        if random.random() < 0.5:
                            end_area = (80, 10, 90, 20)
                        else:
                            end_area = (10, 80, 20, 90)

                        specific_params = TaskSpecificParams(
                            unload_time=50 + random.random() * 100,
                            full_rate=0.1 + random.random() * 0.4
                        )
                    else:
                        if random.random() < 0.5:
                            start_area = (30, 30, 40, 40)
                        else:
                            start_area = (60, 60, 70, 70)

                        if random.random() < 0.5:
                            end_area = (60, 30, 70, 40)
                        else:
                            end_area = (30, 60, 40, 70)

                        specific_params = TaskSpecificParams(
                            empty_rate=0.1 + random.random() * 0.4,
                            load_time=50 + random.random() * 100
                        )

                    # Find valid positions
                    start_pos = self._find_valid_position(start_area)
                    end_pos = self._find_valid_position(end_area)

                    if start_pos and end_pos:
                        # Create task
                        task = Task(
                            id=task_id,
                            type=task_type,
                            state='open',
                            start=start_pos,
                            end=end_pos,
                            open_time=self.global_time,
                            deadline=self.global_time + random.randint(150, 350),
                            specific=specific_params
                        )

                        self.tasks.append(task)
                        self.task_queues.all.append(task_id)
                        self.task_queues.open.append(task_id)
                        new_task_ids.append(task_id)

                        # Mark on map
                        self.map_matrix[start_pos.x, start_pos.y] = 2
                        self.map_matrix[end_pos.x, end_pos.y] = 3

                if new_task_ids:
                    print(f"Time {self.global_time}: Generated {len(new_task_ids)} new tasks "
                          f"(IDs: {min(new_task_ids)}-{max(new_task_ids)})")

    def _find_valid_position(self, area: Tuple[int, int, int, int]) -> Optional[Position]:
        """Find valid position within given area"""
        max_attempts = 10
        attempts = 0

        while attempts < max_attempts:
            x = random.randint(area[0], area[2])
            y = random.randint(area[1], area[3])

            if self.map_matrix[x, y] == 0:
                return Position(x, y)

            attempts += 1

        return None

    def calculate_metrics(self) -> dict:
        """Calculate current system metrics"""
        metrics = {
            'queue_status': {
                'open': len(self.task_queues.open),
                'assigned': len(self.task_queues.assigned),
                'completed': len(self.task_queues.completed),
                'new': len(self.task_queues.new),
                'total': len(self.task_queues.all)
            },
            'waiting_time': {
                'average': 0,
                'max': 0,
                'average_priority': 0,
                'max_priority': 0
            },
            'robot_status': {
                'idle': 0,
                'moving': 0,
                'working': 0
            },
            'performance': {
                'throughput': 0,
                'recent_completions': 0,
                'average_execution_time': 0
            }
        }

        # Calculate waiting time statistics for open tasks
        if self.task_queues.open:
            waiting_times = []
            priorities = []
            for task_id in self.task_queues.open:
                task = self.tasks[task_id - 1]
                waiting_time = self.global_time - task.open_time
                waiting_times.append(waiting_time)
                priorities.append(task.priority)

            metrics['waiting_time'].update({
                'average': np.mean(waiting_times),
                'max': max(waiting_times),
                'average_priority': np.mean(priorities),
                'max_priority': max(priorities)
            })

        # Calculate robot status
        for robot in self.robots:
            metrics['robot_status'][robot.state] += 1

        # Calculate performance metrics
        recent_completions = [
            completion for completion in self.task_completions
            if completion['time'] > self.global_time - 50
        ]

        if recent_completions:
            metrics['performance'].update({
                'throughput': len(recent_completions) / 50,
                'recent_completions': len(recent_completions),
                'average_execution_time': np.mean([c['execution_time'] for c in recent_completions])
            })

        return metrics

    def print_current_state(self):
        """Print current system state"""
        metrics = self.calculate_metrics()

        print(f"\n----- Time: {self.global_time} -----")

        # Print queue status
        total_tasks = metrics['queue_status']['total']
        completed_tasks = metrics['queue_status']['completed']
        completion_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0

        print(f"Task queues: Waiting={metrics['queue_status']['open']}, "
              f"Executing={metrics['queue_status']['assigned']}, "
              f"Completed={completed_tasks}/{total_tasks} ({completion_percentage:.1f}%)")

        # Print waiting time
        print(f"Waiting time: Average={metrics['waiting_time']['average']:.1f}, "
              f"Max={metrics['waiting_time']['max']:.1f}")

        # Print robot status
        robot_status = metrics['robot_status']
        utilization = ((robot_status['moving'] + robot_status['working']) /
                       SimConfig.NUM_ROBOTS * 100)
        print(f"Robots: Idle={robot_status['idle']}, Moving={robot_status['moving']}, "
              f"Working={robot_status['working']}, Utilization={utilization:.1f}%")

    def run(self):
        """Main simulation loop"""
        print("\nStarting simulation...")
        print(f"Initial configuration:")
        print(f"Map size: {SimConfig.MAP_SIZE}x{SimConfig.MAP_SIZE}")
        print(f"Number of robots: {SimConfig.NUM_ROBOTS}")
        print(f"Initial tasks: {SimConfig.NUM_INITIAL_TASKS}")
        print("=================================")

        while not self._check_termination():
            # Time step
            self.global_time += SimConfig.TIME_UNIT

            # Update task priorities at intervals
            if self.global_time % SimConfig.PRIORITY_UPDATE_INTERVAL == 0:
                self.calculate_dynamic_priorities()

            # Update robot status
            self.update_robot_status()

            # Assign tasks
            self.assign_tasks()

            # Print current state at intervals
            if self.global_time % 10 == 0 or self.global_time < 10:
                self.print_current_state()

            # Check for stuck robots
            self._check_stuck_robots()

            # Check task completion status
            is_complete, uncompleted = self.task_tracker.check_task_completion(self.robots)
            if not is_complete:
                print("\nWarning: Not all tasks were completed!")
                print(f"Uncompleted task IDs: {uncompleted}")

        # Print final summary
        self.print_summary()

    def _check_termination(self) -> bool:
        """Check if simulation should terminate"""
        # Check max simulation time
        if self.global_time >= SimConfig.MAX_SIMULATION_TIME:
            return True

        # Check if all tasks completed and all robots idle
        all_tasks_completed = (
                not self.task_queues.open and
                not self.task_queues.assigned and
                not self.task_queues.new
        )

        all_robots_idle = all(robot.state == 'idle' for robot in self.robots)

        return all_tasks_completed and all_robots_idle

    def _check_stuck_robots(self):
        """Check for and reset stuck robots"""
        stuck_robots = []

        for robot in self.robots:
            if (robot.state != 'idle' and
                    self.global_time - robot.last_update_time > 30):
                stuck_robots.append(robot.id)

                print(f"Warning: Robot {robot.id} stuck "
                      f"(State: {robot.state}, Last update: {robot.last_update_time})")

                # Reset robot status
                robot.state = 'idle'
                robot.path = []
                robot.current_task_id = 0
                robot.last_update_time = self.global_time

        if stuck_robots:
            print(f"Reset {len(stuck_robots)} stuck robots")

    def print_summary(self):
        """Print simulation summary"""
        print("\n===== Simulation Summary =====")
        print(f"Total runtime: {self.global_time} time units")

        # 系统整体统计
        total_completed = len(self.task_queues.completed)
        print(f"\nOverall System Statistics:")
        print(f"Total tasks: {SimConfig.NUM_INITIAL_TASKS}")
        print(f"Total completed: {total_completed}")
        print(f"Completion rate: {(total_completed / SimConfig.NUM_INITIAL_TASKS) * 100:.1f}%")

        # 每个机器人的详细统计
        print("\n===== Robot Task Statistics =====")
        for robot in self.robots:
            total_completed, type1_count, type2_count, completed_ids = robot.get_task_statistics(self.tasks)

            print(f"\nRobot {robot.id}:")
            print(f"  Total completed: {total_completed}")
            print(f"  Type 1 (Material Transport) completed: {type1_count}")
            print(f"  Type 2 (Empty Barrel Transport) completed: {type2_count}")
            print(f"  Completed task IDs: {completed_ids}")


def main():
    """Main entry point"""
    # Create and initialize system
    system = System()
    system.initialize()

    # Run simulation
    system.run()


if __name__ == "__main__":
    main()