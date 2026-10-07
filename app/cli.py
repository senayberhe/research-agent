"""Account administration (no sign-up: admins create accounts).

    python -m app.cli create-user alice          # prompts for the password; role viewer
    python -m app.cli create-user bob --role researcher
    python -m app.cli set-role alice admin       # viewer, researcher or admin
    python -m app.cli list-users
    python -m app.cli set-password alice
    python -m app.cli deactivate alice           # can't sign in; signed out everywhere
    python -m app.cli activate alice

In Docker: docker compose exec api uv run python -m app.cli create-user alice
"""

import argparse
import asyncio
import getpass
import sys

from app.db.database import AsyncSessionLocal, engine
from app.services.user_service import (
    UserError,
    create_user,
    list_users,
    set_active,
    set_password,
    set_role,
)


def _read_password(confirm: bool = True) -> str:
    # Piped in (e.g. from a secret): read it as is.
    if not sys.stdin.isatty():
        return sys.stdin.readline().rstrip("\n")

    password = getpass.getpass("Password: ")

    if confirm and getpass.getpass("Repeat password: ") != password:
        raise UserError("The passwords don't match.")

    return password


async def _run(args: argparse.Namespace) -> None:
    # The SQL echo (on in development) would bury this tool's output.
    engine.echo = False

    async with AsyncSessionLocal() as db:

        if args.command == "create-user":
            user = await create_user(db, args.username, _read_password(), args.role)
            print(f"Created user {user.username!r} (id {user.id}, {user.role}).")

        elif args.command == "set-role":
            user = await set_role(db, args.username, args.role)
            print(f"{user.username!r} is now {user.role}.")

        elif args.command == "set-password":
            user = await set_password(db, args.username, _read_password())
            print(f"Password changed for {user.username!r}.")

        elif args.command in ("activate", "deactivate"):
            user = await set_active(db, args.username, args.command == "activate")
            print(f"{user.username!r} is now {'active' if user.is_active else 'deactivated'}.")

        elif args.command == "list-users":
            for user in await list_users(db):
                state = "active" if user.is_active else "deactivated"
                last = user.last_login_at.isoformat() + "Z" if user.last_login_at else "never"
                print(f"{user.id:>4}  {user.username:<30} {user.role:<11} {state:<12} last sign-in: {last}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create-user")
    create.add_argument("username")
    create.add_argument("--role", default="viewer", help="viewer (default), researcher or admin")

    role = commands.add_parser("set-role")
    role.add_argument("username")
    role.add_argument("role", help="viewer, researcher or admin")

    for name in ("set-password", "activate", "deactivate"):
        commands.add_parser(name).add_argument("username")
    commands.add_parser("list-users")

    args = parser.parse_args(argv)

    try:
        asyncio.run(_run(args))
    except UserError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
