import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
import heapq
import time
from collections import defaultdict


@dataclass
class Position:
    x: int
    y: int

    def __eq__(self, other):
        if not isinstance(other, Position):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __lt__(self, other):
        if not isinstance(other, Position):
            return NotImplemented
        return (self.x, self.y) < (other.x, other.y)


@dataclass
class Robot:
    id: int
    start: Position
    goal: Position
    path: List[Position] = None


@dataclass
class Constraint:
    robot: int
    pos: Position
    timestep: int
    prev_pos: Optional[Position] = None


@dataclass
class CBSNode:
    cost: int
    constraints: List[Constraint]
    solution: Dict[int, List[Position]]
    conflicts: List[Tuple]

    def __lt__(self, other):
        return self.cost < other.cost


@dataclass
class PathNode:
    f_score: float
    g_score: float
    pos: Position
    path: List[Position]

    def __lt__(self, other):
        return self.f_score < other.f_score


class LowLevelPlanner:
    def __init__(self, grid_size: int, obstacles: Set[Position]):
        self.grid_size = grid_size
        self.obstacles = obstacles

    def get_neighbors(self, pos: Position) -> List[Position]:
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        neighbors = []

        for dx, dy in directions:
            new_pos = Position(pos.x + dx, pos.y + dy)
            if (0 <= new_pos.x < self.grid_size and
                    0 <= new_pos.y < self.grid_size and
                    new_pos not in self.obstacles):
                neighbors.append(new_pos)

        return neighbors

    def manhattan_distance(self, pos1: Position, pos2: Position) -> int:
        return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)

    def plan_path(self, start: Position, goal: Position,
                  constraints: List[Constraint]) -> List[Position]:
        open_set = []
        initial_node = PathNode(
            f_score=self.manhattan_distance(start, goal),
            g_score=0,
            pos=start,
            path=[start]
        )
        heapq.heappush(open_set, initial_node)
        closed_set = set()

        while open_set:
            current = heapq.heappop(open_set)

            if current.pos == goal:
                return current.path

            if (current.pos, len(current.path) - 1) in closed_set:
                continue

            closed_set.add((current.pos, len(current.path) - 1))

            for next_pos in self.get_neighbors(current.pos):
                timestep = len(current.path)
                constrained = False
                for constraint in constraints:
                    if (constraint.timestep == timestep and
                            constraint.pos == next_pos):
                        constrained = True
                        break

                if constrained:
                    continue

                new_g = current.g_score + 1
                new_f = new_g + self.manhattan_distance(next_pos, goal)
                new_path = current.path + [next_pos]

                new_node = PathNode(
                    f_score=new_f,
                    g_score=new_g,
                    pos=next_pos,
                    path=new_path
                )
                heapq.heappush(open_set, new_node)

        return []


class CBSSolver:
    def __init__(self, grid_size: int, obstacles: Set[Position]):
        self.grid_size = grid_size
        self.obstacles = obstacles
        self.low_level = LowLevelPlanner(grid_size, obstacles)

    def get_first_conflict(self, solution: Dict[int, List[Position]]) -> Optional[Tuple]:
        for r1 in solution:
            path1 = solution[r1]
            for r2 in solution:
                if r1 >= r2:
                    continue

                path2 = solution[r2]
                length = min(len(path1), len(path2))

                for t in range(length):
                    if path1[t] == path2[t]:
                        return ('vertex', r1, r2, path1[t], t)

                for t in range(length - 1):
                    if path1[t] == path2[t + 1] and path1[t + 1] == path2[t]:
                        return ('edge', r1, r2, path1[t], path1[t + 1], t)

        return None

    def get_sum_of_costs(self, solution: Dict[int, List[Position]]) -> int:
        return sum(len(path) for path in solution.values())

    def solve(self, robots: List[Robot], max_iterations: int = 1000) -> Dict[int, List[Position]]:
        initial_solution = {}
        initial_constraints = []

        for robot in robots:
            path = self.low_level.plan_path(robot.start, robot.goal, [])
            if not path:
                return None
            initial_solution[robot.id] = path

        root = CBSNode(
            cost=self.get_sum_of_costs(initial_solution),
            constraints=initial_constraints,
            solution=initial_solution,
            conflicts=[]
        )

        open_list = [root]
        iterations = 0

        while open_list and iterations < max_iterations:
            iterations += 1
            node = heapq.heappop(open_list)

            conflict = self.get_first_conflict(node.solution)
            if not conflict:
                print(f"Solution found after {iterations} iterations")
                return node.solution

            if conflict[0] == 'vertex':
                _, r1, r2, pos, t = conflict
                for robot_id in (r1, r2):
                    new_constraints = node.constraints + [
                        Constraint(robot=robot_id, pos=pos, timestep=t)
                    ]

                    robot = next(r for r in robots if r.id == robot_id)
                    new_path = self.low_level.plan_path(
                        robot.start, robot.goal, new_constraints
                    )

                    if new_path:
                        new_solution = node.solution.copy()
                        new_solution[robot_id] = new_path

                        new_node = CBSNode(
                            cost=self.get_sum_of_costs(new_solution),
                            constraints=new_constraints,
                            solution=new_solution,
                            conflicts=[]
                        )
                        heapq.heappush(open_list, new_node)

            elif conflict[0] == 'edge':
                _, r1, r2, pos1, pos2, t = conflict
                for robot_id, pos_a, pos_b in [(r1, pos1, pos2), (r2, pos2, pos1)]:
                    new_constraints = node.constraints + [
                        Constraint(robot=robot_id, pos=pos_b, timestep=t + 1,
                                   prev_pos=pos_a)
                    ]

                    robot = next(r for r in robots if r.id == robot_id)
                    new_path = self.low_level.plan_path(
                        robot.start, robot.goal, new_constraints
                    )

                    if new_path:
                        new_solution = node.solution.copy()
                        new_solution[robot_id] = new_path

                        new_node = CBSNode(
                            cost=self.get_sum_of_costs(new_solution),
                            constraints=new_constraints,
                            solution=new_solution,
                            conflicts=[]
                        )
                        heapq.heappush(open_list, new_node)

        print(f"No solution found after {max_iterations} iterations")
        return None


class TestEnvironment:
    MAP_SIZE = 100
    NUM_ROBOTS = 10
    NUM_TASKS = 45

    def __init__(self):
        self.obstacles = set()
        self.material_load_areas = [
            (10, 10, 20, 20),
            (80, 80, 90, 90)
        ]
        self.material_unload_areas = [
            (80, 10, 90, 20),
            (10, 80, 20, 90)
        ]
        self.empty_load_areas = [
            (30, 30, 40, 40),
            (60, 60, 70, 70)
        ]
        self.empty_unload_areas = [
            (60, 30, 70, 40),
            (30, 60, 40, 70)
        ]

        self.create_map()
        self.robots = self.initialize_robots()
        self.tasks = self.generate_tasks()

    def create_map(self):
        num_obstacles = round(self.MAP_SIZE * self.MAP_SIZE * 0.10)

        obstacles_created = 0
        while obstacles_created < num_obstacles:
            x = np.random.randint(5, self.MAP_SIZE - 5)
            y = np.random.randint(5, self.MAP_SIZE - 5)

            if Position(x, y) not in self.obstacles:
                self.obstacles.add(Position(x, y))
                for dx in range(min(3, self.MAP_SIZE - x)):
                    for dy in range(min(3, self.MAP_SIZE - y)):
                        self.obstacles.add(Position(x + dx, y + dy))
                obstacles_created += 1

        num_corridors = 4
        for _ in range(num_corridors):
            y = np.random.randint(10, self.MAP_SIZE - 10)
            for x in range(5, self.MAP_SIZE - 5):
                self.obstacles.discard(Position(x, y))
                self.obstacles.discard(Position(x, y + 1))

            x = np.random.randint(10, self.MAP_SIZE - 10)
            for y in range(5, self.MAP_SIZE - 5):
                self.obstacles.discard(Position(x, y))
                self.obstacles.discard(Position(x + 1, y))

    def find_valid_position(self, area: Tuple[int, int, int, int]) -> Position:
        max_attempts = 10
        for _ in range(max_attempts):
            x = np.random.randint(area[0], area[2])
            y = np.random.randint(area[1], area[3])
            pos = Position(x, y)
            if pos not in self.obstacles:
                return pos
        return None

    def initialize_robots(self) -> List[Robot]:
        robots = []
        for i in range(self.NUM_ROBOTS):
            while True:
                x = np.random.randint(5, self.MAP_SIZE - 5)
                y = np.random.randint(5, self.MAP_SIZE - 5)
                pos = Position(x, y)
                if pos not in self.obstacles and not any(r.start == pos for r in robots):
                    robots.append(Robot(id=i + 1, start=pos, goal=pos))
                    break
        return robots

    def generate_tasks(self) -> List[Tuple[Position, Position, int]]:
        tasks = []
        num_type1 = self.NUM_TASKS // 2

        for i in range(self.NUM_TASKS):
            task_type = 1 if i < num_type1 else 2

            if task_type == 1:
                start_area = self.material_load_areas[np.random.randint(0, len(self.material_load_areas))]
                end_area = self.material_unload_areas[np.random.randint(0, len(self.material_unload_areas))]
            else:
                start_area = self.empty_load_areas[np.random.randint(0, len(self.empty_load_areas))]
                end_area = self.empty_unload_areas[np.random.randint(0, len(self.empty_unload_areas))]

            start_pos = self.find_valid_position(start_area)
            end_pos = self.find_valid_position(end_area)

            if start_pos and end_pos:
                tasks.append((start_pos, end_pos, task_type))

        return tasks


def test_cbs_with_tasks():
    print("\n=============== CBS算法测试开始 ===============")
    print("正在初始化测试环境...")
    env = TestEnvironment()

    print("\n【环境配置】")
    print(f"地图大小: {env.MAP_SIZE}x{env.MAP_SIZE}")
    print(f"机器人数量: {env.NUM_ROBOTS}")
    print(f"任务总数: {env.NUM_TASKS}")
    print(f"障碍物总数: {len(env.obstacles)}")

    print("\n【初始机器人位置】")
    for robot in env.robots:
        print(f"机器人 {robot.id}: ({robot.start.x}, {robot.start.y})")

    print("\n【工作区域】")
    print("材料装载区:")
    for area in env.material_load_areas:
        print(f"  区域: ({area[0]}, {area[1]}) -> ({area[2]}, {area[3]})")
    print("材料卸载区:")
    for area in env.material_unload_areas:
        print(f"  区域: ({area[0]}, {area[1]}) -> ({area[2]}, {area[3]})")
    print("空桶装载区:")
    for area in env.empty_load_areas:
        print(f"  区域: ({area[0]}, {area[1]}) -> ({area[2]}, {area[3]})")
    print("空桶卸载区:")
    for area in env.empty_unload_areas:
        print(f"  区域: ({area[0]}, {area[1]}) -> ({area[2]}, {area[3]})")

    solver = CBSSolver(env.MAP_SIZE, env.obstacles)

    stats = {
        'total_makespan': 0,
        'total_path_length': 0,
        'completed_tasks': 0,
        'type1_completed': 0,
        'type2_completed': 0,
        'robot_stats': defaultdict(lambda: {
            'tasks_completed': 0,
            'type1_completed': 0,
            'type2_completed': 0,
            'total_distance': 0,
            'paths': []
        })
    }

    start_time = time.time()

    print("\n【任务执行详情】")
    for task_idx, (start_pos, end_pos, task_type) in enumerate(env.tasks, 1):
        print(f"\n=== 任务 {task_idx}/{env.NUM_TASKS} ===")
        print(f"类型: {'材料运输' if task_type == 1 else '空桶运输'}")
        print(f"起点: ({start_pos.x}, {start_pos.y})")
        print(f"终点: ({end_pos.x}, {end_pos.y})")

        available_robots = [r for r in env.robots if r.goal == r.start]
        if not available_robots:
            print("错误: 没有可用的机器人")
            continue

        robot = min(available_robots,
                    key=lambda r: solver.low_level.manhattan_distance(r.start, start_pos))
        print(f"分配给机器人 {robot.id} (当前位置: ({robot.start.x}, {robot.start.y}))")

        path_to_start = solver.low_level.plan_path(robot.start, start_pos, [])
        if not path_to_start:
            print("错误: 无法规划到起点的路径")
            continue

        path_to_end = solver.low_level.plan_path(start_pos, end_pos, [])
        if not path_to_end:
            print("错误: 无法规划到终点的路径")
            continue

        full_path = path_to_start + path_to_end[1:]
        path_points = [(p.x, p.y) for p in full_path]

        path_length = len(full_path) - 1
        stats['total_path_length'] += path_length
        stats['total_makespan'] += path_length
        stats['completed_tasks'] += 1
        if task_type == 1:
            stats['type1_completed'] += 1
        else:
            stats['type2_completed'] += 1

        robot_stat = stats['robot_stats'][robot.id]
        robot_stat['tasks_completed'] += 1
        robot_stat['total_distance'] += path_length
        if task_type == 1:
            robot_stat['type1_completed'] += 1
        else:
            robot_stat['type2_completed'] += 1
        robot_stat['paths'].append(path_points)

        robot.start = end_pos
        robot.goal = end_pos

        print(f"路径长度: {path_length}")
        print(f"详细路径: {path_points}")

    end_time = time.time()
    execution_time = end_time - start_time

    print("\n=============== 测试结果总结 ===============")
    print(f"\n【执行时间统计】")
    print(f"总运行时间: {execution_time:.2f} 秒")
    print(f"平均每个任务时间: {execution_time / env.NUM_TASKS:.2f} 秒")

    print(f"\n【任务完成统计】")
    completion_rate = (stats['completed_tasks'] / env.NUM_TASKS) * 100
    print(f"总任务数: {env.NUM_TASKS}")
    print(f"完成任务数: {stats['completed_tasks']}")
    print(f"完成率: {completion_rate:.2f}%")
    print(f"类型1(材料运输)完成数: {stats['type1_completed']}")
    print(f"类型2(空桶运输)完成数: {stats['type2_completed']}")

    print(f"\n【路径统计】")
    if stats['completed_tasks'] > 0:
        print(f"总路径长度: {stats['total_path_length']}")
        print(f"平均路径长度: {stats['total_path_length'] / stats['completed_tasks']:.2f}")
        print(f"平均Makespan: {stats['total_makespan'] / stats['completed_tasks']:.2f}")

    print(f"\n【机器人详细统计】")
    for robot_id, robot_stat in sorted(stats['robot_stats'].items()):
        print(f"\n机器人 {robot_id}:")
        print(f"  完成任务总数: {robot_stat['tasks_completed']}")
        print(f"  类型1任务数: {robot_stat['type1_completed']}")
        print(f"  类型2任务数: {robot_stat['type2_completed']}")
        print(f"  总行驶距离: {robot_stat['total_distance']}")
        if robot_stat['tasks_completed'] > 0:
            print(f"  平均每任务距离: {robot_stat['total_distance'] / robot_stat['tasks_completed']:.2f}")
        print(f"  执行的所有路径:")
        for idx, path in enumerate(robot_stat['paths'], 1):
            print(f"    任务{idx}: {path}")

    print("\n【未分配机器人】")
    unused_robots = [r for r in env.robots if r.id not in stats['robot_stats']]
    if unused_robots:
        print(f"以下机器人未被使用:")
        for robot in unused_robots:
            print(f"  机器人 {robot.id}: 始终位于 ({robot.start.x}, {robot.start.y})")
    else:
        print("所有机器人都参与了任务")

    print("\n=============== CBS算法测试结束 ===============")


if __name__ == "__main__":
    test_cbs_with_tasks()