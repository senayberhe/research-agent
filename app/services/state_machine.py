from app.db.models import (
    AgentRun,
    AgentRunStatus,
    ResearchTask,
    TaskStatus,
)


class InvalidStateTransition(Exception):
    """Raised when a task or run is moved to a status it can't reach."""


# Where each task status may go next.
#
#   PENDING → PLANNING → RESEARCHING → SYNTHESIZING → COMPLETED
#      └──────────┴───────────┴─────────────┴──→ FAILED
#                                                  │
#             RESEARCHING ←──── (resume) ──────────┘
#
# COMPLETED is final. FAILED is not: a failed task can be resumed, which
# goes straight back to RESEARCHING.
TASK_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.PLANNING, TaskStatus.FAILED},
    TaskStatus.PLANNING: {TaskStatus.RESEARCHING, TaskStatus.FAILED},
    TaskStatus.RESEARCHING: {TaskStatus.SYNTHESIZING, TaskStatus.FAILED},
    TaskStatus.SYNTHESIZING: {TaskStatus.COMPLETED, TaskStatus.FAILED},
    TaskStatus.COMPLETED: set(),
    TaskStatus.FAILED: {TaskStatus.RESEARCHING},
    # Unused legacy status; kept so old rows still load.
    TaskStatus.IN_PROGRESS: {TaskStatus.FAILED},
}


# Where each agent run status may go next.
#
#   CREATED → RUNNING → WAITING_FOR_TOOL → PROCESSING_RESULT ─┐
#               ▲  │                                          │
#               │  └──→ COMPLETED                             │
#               └─────────────────────────────────────────────┘
#
#   Any unfinished status → FAILED → (resume) → RUNNING
#
# COMPLETED is final. A resume reopens the same run, so the run's tokens,
# counts and checkpoint cover the whole job.
AGENT_RUN_TRANSITIONS: dict[AgentRunStatus, set[AgentRunStatus]] = {
    AgentRunStatus.CREATED: {
        AgentRunStatus.RUNNING,
        AgentRunStatus.FAILED,
    },
    AgentRunStatus.RUNNING: {
        AgentRunStatus.WAITING_FOR_TOOL,
        AgentRunStatus.COMPLETED,
        AgentRunStatus.FAILED,
    },
    AgentRunStatus.WAITING_FOR_TOOL: {
        AgentRunStatus.PROCESSING_RESULT,
        AgentRunStatus.FAILED,
    },
    AgentRunStatus.PROCESSING_RESULT: {
        AgentRunStatus.RUNNING,
        AgentRunStatus.FAILED,
    },
    AgentRunStatus.COMPLETED: set(),
    AgentRunStatus.FAILED: {
        AgentRunStatus.RUNNING,
    },
}


def can_transition_task(
    task: ResearchTask,
    new_status: TaskStatus,
) -> bool:

    # status is a plain string column, so a loaded row holds "failed" rather
    # than TaskStatus.FAILED; TaskStatus(...) accepts both.
    return new_status in TASK_TRANSITIONS[TaskStatus(task.status)]


def transition_task(
    task: ResearchTask,
    new_status: TaskStatus,
) -> ResearchTask:

    if not can_transition_task(task, new_status):
        raise InvalidStateTransition(
            f"Research task {task.id} can't go from "
            f"{TaskStatus(task.status).value} to {new_status.value}"
        )

    task.status = new_status

    return task


def can_transition_agent_run(
    run: AgentRun,
    new_status: AgentRunStatus,
) -> bool:

    return new_status in AGENT_RUN_TRANSITIONS[AgentRunStatus(run.status)]


def transition_agent_run(
    run: AgentRun,
    new_status: AgentRunStatus,
) -> AgentRun:

    current = AgentRunStatus(run.status)

    if not can_transition_agent_run(run, new_status):
        raise InvalidStateTransition(
            f"Agent run {run.id} can't go from "
            f"{current.value} to {new_status.value}"
        )

    run.status = new_status

    return run


def is_resumable(task: ResearchTask) -> bool:
    """A task can be resumed if it failed."""

    return TaskStatus(task.status) == TaskStatus.FAILED
