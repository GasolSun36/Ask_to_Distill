import json
import multiprocessing
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
import os
from loguru import logger
import time
from tau2.agent.llm_agent import LLMAgent, LLMGTAgent, LLMSoloAgent
from tau2.data_model.simulation import (
    AgentInfo,
    Info,
    Results,
    RunConfig,
    SimulationRun,
    UserInfo,
)
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment, EnvironmentInfo
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.metrics.agent_metrics import compute_metrics
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import RegistryInfo, registry
from tau2.user.user_simulator import DummyUser, get_global_user_sim_guidelines
from tau2.utils.display import ConsoleDisplay
from tau2.utils.pydantic_utils import get_pydantic_hash
from tau2.utils.utils import DATA_DIR, get_commit_hash, get_now, show_dict_diff


def get_options() -> RegistryInfo:
    """
    Returns options for the simulator.
    """
    return registry.get_info()


def get_environment_info(
    domain_name: str, include_tool_info: bool = False
) -> EnvironmentInfo:
    """Get information about the environment for a registered Domain"""
    global registry
    env_constructor = registry.get_env_constructor(domain_name)
    # print(44,env_constructor)
    # exit(0)
    return env_constructor().get_info(include_tool_info=include_tool_info)


def load_tasks(task_set_name: str, task_path: str, save_to) -> list[Task]:
    """
    Loads the tasks for the given domain.
    """
    global registry
    task_loader = registry.get_tasks_loader(task_set_name)
    tasks = task_loader(task_path=task_path,save_to=save_to)
    return tasks


def get_tasks(
    task_set_name: str,
    task_ids: Optional[list[str]] = None,
    num_tasks: Optional[int] = None,
    task_path = '',
    save_to = ''
) -> list[Task]:
    """
    Loads the tasks for the given domain.
    """
    if task_ids is None:
        return load_tasks(task_set_name=task_set_name,task_path=task_path,save_to=save_to)
    tasks = [
        task for task in load_tasks(task_set_name=task_set_name,task_path=task_path) if task.id in task_ids
    ]
    if len(tasks) != len(task_ids):
        missing_tasks = set(task_ids) - set([task.id for task in tasks])
        raise ValueError(
            f"Not all tasks were found for task set {task_set_name}: {missing_tasks}"
        )
    if num_tasks is not None:
        tasks = tasks[:num_tasks]
    return tasks


def make_run_name(config: RunConfig) -> str:
    """
    Make a run name from the run config
    """
    clean_llm_agent_name = config.llm_agent.split("/")[-1]
    agent_name = f"{config.agent}_{clean_llm_agent_name}"

    clean_llm_user_name = config.llm_user.split("/")[-1]
    user_name = f"{config.user}_{clean_llm_user_name}"

    return f"{get_now()}_{config.domain}_{agent_name}_{user_name}"


def run_domain(config: RunConfig) -> Results:
    """
    Run simulations for a domain
    """
    config.validate()
    # ConsoleDisplay.display_run_config(config)
    if config.task_set_name is None:
        task_set_name = config.domain
    else:
        task_set_name = config.task_set_name
    tasks = get_tasks(task_set_name, config.task_ids, config.num_tasks, task_path=config.task_path, save_to=config.output_file)
    # print(104,'tasks',tasks)
    # exit(0)
    # 104 config.agent llm_agent
    if "gt" in config.agent:
        total_num_tasks = len(tasks)
        tasks = [task for task in tasks if LLMGTAgent.check_valid_task(task)]
        num_tasks = len(tasks)
        ConsoleDisplay.console.print(
            f"[bold green]Running {num_tasks} out of {total_num_tasks} tasks for GT agent.[/bold green]"
        )
    if "solo" in config.agent:
        total_num_tasks = len(tasks)
        tasks = [task for task in tasks if LLMSoloAgent.check_valid_task(task)]
        num_tasks = len(tasks)
        ConsoleDisplay.console.print(
            f"[bold green]Running {num_tasks} out of {total_num_tasks} tasks for solo agent.[/bold green]"
        )

    num_trials = config.num_trials
    save_to = config.save_to
    if save_to is None:
        save_to = make_run_name(config)
    save_to = config.output_file
    # print('save path:',save_to)
    logger.info(f"[INFO] Starting run_tasks for domain={config.domain}, tasks={len(tasks)}")
    simulation_results = run_tasks(
        domain=config.domain,
        tasks=tasks,
        agent=config.agent,
        user=config.user,
        llm_agent=config.llm_agent,
        llm_args_agent=config.llm_args_agent,
        llm_user=config.llm_user,
        llm_args_user=config.llm_args_user,
        num_trials=num_trials,
        max_steps=config.max_steps,
        max_errors=config.max_errors,
        save_to=save_to,
        console_display=True,
        evaluation_type=EvaluationType.ALL,
        max_concurrency=config.max_concurrency,
        seed=config.seed,
        log_level=config.log_level,
        cur_transfer_dir=config.cur_transfer_dir,
        model_config_path=config.model_config_path,
        use_model_tool=config.use_model_tool
    )
    # metrics = compute_metrics(simulation_results)
    # ConsoleDisplay.display_agent_metrics(metrics)

    return simulation_results


def run_tasks(
    domain: str,
    tasks: list[Task],
    agent: str,
    user: str,
    llm_agent: Optional[str] = None,
    llm_args_agent: Optional[dict] = None,
    llm_user: Optional[str] = None,
    llm_args_user: Optional[dict] = None,
    num_trials: int = 1,
    max_steps: int = 100,
    max_errors: int = 10,
    save_to: Optional[str | Path] = None,
    console_display: bool = True,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    max_concurrency: int = 1,
    seed: Optional[int] = 300,
    log_level: Optional[str] = "INFO",
    cur_transfer_dir: str = '',
    model_config_path: str = '',
    use_model_tool: bool = False
) -> Results:
    """
    Runs tasks for a given domain.
    If llm_as_judge is True, the LLM will be used to annotate the simulation run.
    Calculates the reward for the simulation run.
    Args:
        domain (str): The domain to run the simulation on.
        tasks (list[Task]): The tasks to run.
        agent (str): The agent to run the simulation on.
        user (str): The user to run the simulation on.
        llm_agent (str): The model to use for the agent.
        llm_args_agent (dict): The arguments to pass to the LLM for the agent.
        llm_user (str): The model to use for the user.
        llm_args_user (dict): The arguments to pass to the LLM for the user.
        max_steps (int): The maximum number of steps to run the simulation.
        max_errors (int): The maximum number of errors to allow in the simulation.
        save_to (str | Path): The path to json file where to save the simulation results. If the file already exists, it will try to resume the run.
        evaluation_type (EvaluationType): The type of evaluation to use.
        max_concurrency (int): The maximum number of concurrent simulations to run.
        seed (int): The seed to use for the simulation.
        log_level (str): The log level to use.
    Returns:
        The simulation results and the annotations (if llm_review is True).
    """
    if isinstance(save_to, str):
        save_to = Path(save_to)
    save_dir = str(save_to)
    assert save_dir.endswith('.json'), f"save_dir must end with .json, but got {save_dir}"
    save_dir = save_dir[:-len('.json')]
    logger.info(f"[INFO] save_dir={save_dir}")
    # updated_tasks = []

    # Set log level from config
    logger.info(f"[INFO] remove logger started, log_level={log_level}")
    logger.remove()
    logger.add(lambda msg: print(msg, end="", flush=True), level=log_level)
    logger.info(f"[INFO] Logger reconfigured with level={log_level} completed")
    if len(tasks) == 0:
        raise ValueError("No tasks to run")
    if num_trials <= 0:
        raise ValueError("Number of trials must be greater than 0")
    if max_steps <= 0:
        raise ValueError("Max steps must be greater than 0")
    if max_errors <= 0:
        raise ValueError("Max errors must be greater than 0")
    random.seed(seed)
    seeds = [random.randint(0, 1000000) for _ in range(num_trials)]
    if "seed" in llm_args_agent:
        logger.warning("Each trial will modify the seed for the agent")
    if "seed" in llm_args_user:
        logger.warning("Each trial will modify the seed for the user")

    lock = multiprocessing.Lock()
    info = get_info(
        domain=domain,
        agent=agent,
        user=user,
        llm_agent=llm_agent,
        llm_args_agent=llm_args_agent,
        llm_user=llm_user,
        llm_args_user=llm_args_user,
        num_trials=num_trials,
        max_steps=max_steps,
        max_errors=max_errors,
        seed=seed,
    )
    logger.info(f"[INFO] get_info completed")
    simulation_results = Results(
        info=info,
        tasks=tasks,
        simulations=[],
    )
    done_runs = set()
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if save_to is not None:
        # If save_to already exists, check if the user wants to resume the run.
        if save_to.exists():
            os.remove(save_to)
        if not save_to.parent.exists():
            save_to.parent.mkdir(parents=True, exist_ok=True)
        # with open(save_to, "w") as fp:
        #     fp.write(simulation_results.model_dump_json(indent=2))

    def _save(simulation: SimulationRun,latency):
        # print(268,'save')
        if save_to is None:
            return
        cur_simulation = simulation.model_dump()
        with open(os.path.join(save_dir,cur_simulation['id']+'.json'),'w') as f:
            json.dump({
                'reward': cur_simulation["reward_info"]['reward'],
                'id': cur_simulation['id'],
                'task_id': cur_simulation['task_id'],
                'timestamp': cur_simulation['timestamp'],
                'duration': cur_simulation['duration'],
                'agent_cost': cur_simulation['agent_cost'],
                'user_cost': cur_simulation['user_cost'],
                'messages': cur_simulation['messages'],
            }, f, indent=2)
            

    def _run(task: Task, trial: int, seed: int, progress_str: str) -> SimulationRun:
        logger.info(f"[INFO] _run started: task_id={task.id}, trial={trial}, progress={progress_str}")
        start_time = time.time()
        logger.info(f"[INFO] Calling run_task for task_id={task.id}")
        simulation = run_task(
            domain=domain,
            task=task,
            agent=agent,
            user=user,
            llm_agent=llm_agent,
            llm_args_agent=llm_args_agent,
            llm_user=llm_user,
            llm_args_user=llm_args_user,
            max_steps=max_steps,
            max_errors=max_errors,
            evaluation_type=evaluation_type,
            seed=seed,
            cur_transfer_dir=cur_transfer_dir,
            model_config_path=model_config_path,
            use_model_tool=use_model_tool
        )
        latency = time.time()-start_time
        logger.info(f"[INFO] run_task completed for task_id={task.id}, latency={latency:.2f}s")
        simulation.trial = trial
        _save(simulation,latency=latency)

        return simulation

    args = []
    for trial in range(num_trials):
        for i, task in enumerate(tasks):
            if (trial, task.id, seeds[trial]) in done_runs:
                ConsoleDisplay.console.print(
                    f"[bold yellow]Skipping task {task.id}, trial {trial} because it has already been run.[/bold yellow]"
                )
                continue
            progress_str = f"{i}/{len(tasks)} (trial {trial + 1}/{num_trials})"
            args.append((task, trial, seeds[trial], progress_str))

    # res = list(map(_run, *zip(*args)))
    # if res:
    #     simulation_results.simulations.extend(res)
    # print(309,'len(simulation_results.simulations)', len(simulation_results.simulations))
    logger.info(f"[INFO] Starting ThreadPoolExecutor with max_workers={max_concurrency}, total_tasks={len(args)}")
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        logger.info(f"[INFO] Submitting {len(args)} tasks to executor")
        res = list(executor.map(_run, *zip(*args)))
        logger.info(f"[INFO] All tasks completed, got {len(res)} results")
        if res:
            simulation_results.simulations.extend(res)
        print(len(simulation_results.simulations))
    # ConsoleDisplay.console.print(
    #     "\n✨ [bold green]Successfully completed all simulations![/bold green]\n"
    #     "To review the simulations, run: [bold blue]tau2 view[/bold blue]"
    # )
    return simulation_results


def run_task(
    domain: str,
    task: Task,
    agent: str,
    user: str,
    llm_agent: Optional[str] = None,
    llm_args_agent: Optional[dict] = None,
    llm_user: Optional[str] = None,
    llm_args_user: Optional[dict] = None,
    max_steps: int = 100,
    max_errors: int = 10,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    seed: Optional[int] = None,
    cur_transfer_dir: str = '',
    model_config_path: str = '',
    use_model_tool: bool = False
) -> SimulationRun:
    """
    Runs tasks for a given domain.
     If llm_as_judge is True, the LLM will be used to annotate the simulation run.
     Calculates the reward for the simulation run.
     Args:
         domain (str): The domain to run the simulation on.
         task (Task): The task to run.
         agent (str): The agent to run the simulation on.
         user (str): The user to run the simulation on.
         llm_agent (str): The model to use for the agent.
         llm_args_agent (dict): The arguments to pass to the LLM for the agent.
         llm_user (str): The model to use for the user.
         llm_args_user (dict): The arguments to pass to the LLM for the user.
         max_steps (int): The maximum number of steps to run the simulation.
         max_errors (int): The maximum number of errors to allow in the simulation.
         evaluation_type (EvaluationType): The type of evaluation to use.
         seed (int): The seed to use for the simulation.
     Returns:
         The simulation run.
    """

    if max_steps <= 0:
        raise ValueError("Max steps must be greater than 0")
    if max_errors <= 0:
        raise ValueError("Max errors must be greater than 0")
    # if not os.path.isdir(cur_transfer_dir):
    #     os.makedirs(cur_transfer_dir,exist_ok=True)
    global registry
    logger.info(
        f"STARTING SIMULATION: Domain: {domain}, Task: {task.id}, Agent: {agent}, User: {user}"
    )
    logger.info(f"[INFO] run_task: Getting environment constructor for domain={domain}")
    environment_constructor = registry.get_env_constructor(domain)
    environment = environment_constructor()
    AgentConstructor = registry.get_agent_constructor(agent)

    solo_mode = False
    if issubclass(AgentConstructor, LLMAgent):
        # agent class here
        agent = AgentConstructor(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            llm=llm_agent,
            llm_args=llm_args_agent,
            cur_transfer_dir=cur_transfer_dir,
            use_model_tool=use_model_tool,
            model_config_path=model_config_path,
            domain=domain
        )
    elif issubclass(AgentConstructor, LLMGTAgent):
        agent = AgentConstructor(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            llm=llm_agent,
            llm_args=llm_args_agent,
            task=task,
        )
    elif issubclass(AgentConstructor, LLMSoloAgent):
        solo_mode = True
        environment: Environment = environment_constructor(solo_mode=True)
        user_tools = environment.get_user_tools() if environment.user_tools else []
        agent = AgentConstructor(
            tools=environment.get_tools() + user_tools,
            domain_policy=environment.get_policy(),
            llm=llm_agent,
            llm_args=llm_args_agent,
            task=task,
        )
    else:
        raise ValueError(
            f"Unknown agent type: {AgentConstructor}. Should be LLMAgent or LLMSoloAgent"
        )
    try:
        user_tools = environment.get_user_tools()
    except Exception:
        user_tools = None

    UserConstructor = registry.get_user_constructor(user)
    if issubclass(UserConstructor, DummyUser):
        assert isinstance(agent, LLMSoloAgent), (
            "Dummy user can only be used with solo agent"
        )

    logger.info(f"[INFO] run_task: Creating user with UserConstructor={UserConstructor.__name__}")
    user = UserConstructor(
        tools=user_tools,
        instructions=str(task.user_scenario),
        llm=llm_user,
        llm_args=llm_args_user,
    )

    logger.info(f"[INFO] run_task: Creating Orchestrator")
    orchestrator = Orchestrator(
        domain=domain,
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=max_steps,
        max_errors=max_errors,
        seed=seed,
        solo_mode=solo_mode,
        cur_transfer_dir=cur_transfer_dir,
        model_config_path=model_config_path,
        use_model_tool=use_model_tool,
    )
    logger.info(f"[INFO] run_task: Orchestrator created, starting run()")
    simulation = orchestrator.run()

    logger.info(f"[INFO] run_task: Starting evaluate_simulation")
    reward_info = evaluate_simulation(
        domain=domain,
        task=task,
        simulation=simulation,
        evaluation_type=evaluation_type,
        solo_mode=solo_mode,
    )
    logger.info(f"[INFO] run_task: evaluate_simulation completed, reward={reward_info.reward}")

    simulation.reward_info = reward_info
    logger.info(f"[INFO] simulation.reward_info={simulation.reward_info}")

    logger.info(
        f"FINISHED SIMULATION: Domain: {domain}, Task: {task.id}, Agent: {agent.__class__.__name__}, User: {user.__class__.__name__}. Reward: {reward_info.reward}"
    )
    return simulation


def get_info(
    domain: str,
    agent: str,
    user: str,
    llm_agent: Optional[str] = None,
    llm_args_agent: Optional[dict] = None,
    llm_user: Optional[str] = None,
    llm_args_user: Optional[dict] = None,
    num_trials: int = 1,
    max_steps: int = 100,
    max_errors: int = 10,
    seed: Optional[int] = None,
) -> Info:
    user_info = UserInfo(
        implementation=user,
        llm=llm_user,
        llm_args=llm_args_user,
        global_simulation_guidelines=get_global_user_sim_guidelines(),
    )
    agent_info = AgentInfo(
        implementation=agent,
        llm=llm_agent,
        llm_args=llm_args_agent,
    )
    environment_info = get_environment_info(
        domain, include_tool_info=False
    )  # NOTE: Not saving tool info to avoid clutter.
    return Info(
        git_commit='none',
        num_trials=num_trials,
        max_steps=max_steps,
        max_errors=max_errors,
        user_info=user_info,
        agent_info=agent_info,
        environment_info=environment_info,
        seed=seed,
    )
