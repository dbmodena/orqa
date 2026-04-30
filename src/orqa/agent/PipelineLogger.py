from datetime import datetime


RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

BLACK  = "\033[30m"
WHITE  = "\033[97m"

BG_BLUE    = "\033[44m"
BG_GREEN   = "\033[42m"
BG_YELLOW  = "\033[43m"
BG_RED     = "\033[41m"
BG_CYAN    = "\033[46m"
BG_MAGENTA = "\033[45m"
BG_GRAY    = "\033[100m"

CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"


def _ts() -> str:
    return DIM + datetime.now().strftime("%H:%M:%S") + RESET


def _badge(text: str, bg: str) -> str:
    return f"{BOLD}{bg}{BLACK} {text} {RESET}"


def _divider(char: str = "─", width: int = 70) -> str:
    return DIM + char * width + RESET


def _indent(text: str, level: int = 1) -> str:
    pad = "  " * level
    return "\n".join(pad + line for line in str(text).splitlines())


class PipelineLogger:
    """
    Drop-in pretty logger for the query generation pipeline.

    Usage
    -----
    from pipeline_logger import PipelineLogger
    log = PipelineLogger()

    log.section("Single-table generation — sales_2024.csv")
    log.step1_generated(queries)
    log.step2_start(iteration=1)
    log.validator_result(queries_in, queries_out)
    log.judge_result(approved, rejected, failures)
    log.query_rejected(query_id, question, feedback, suggestions)
    log.query_approved(query_id, question)
    log.step3_summary(approved_queries)
    """

    def section(self, title: str) -> None:
        print()
        print(_divider("═"))
        print(f"  {BOLD}{CYAN}{title}{RESET}")
        print(_divider("═"))

    def step(self, number: int, label: str) -> None:
        badge = _badge(f"STEP {number}", BG_CYAN)
        print(f"\n{_ts()}  {badge}  {BOLD}{label}{RESET}")
        print(_indent(_divider(), 1))

    # ------------------------------------------------------------------ #
    # Step 1 — initial generation                                          #
    # ------------------------------------------------------------------ #

    def step1_generated(self, queries: list) -> None:
        self.step(1, "Initial generation — StatementClient")
        if not queries:
            print(_indent(f"{RED}⚠  No queries returned — aborting.{RESET}", 1))
            return
        print(_indent(f"{GREEN}✔  {len(queries)} quer{'y' if len(queries) == 1 else 'ies'} generated{RESET}", 1))
        for q in queries:
            print(_indent(f"{DIM}#{q.get('id', '?')}{RESET}  {q.get('question', '(no question)')}", 2))

    # ------------------------------------------------------------------ #
    # Step 2 — Validator ↔ Judge loop                                     #
    # ------------------------------------------------------------------ #

    def step2_start(self, iteration: int) -> None:
        badge = _badge(f"STEP 2 · iter {iteration}", BG_MAGENTA)
        print(f"\n{_ts()}  {badge}  {BOLD}Validator ↔ Judge loop{RESET}")
        print(_indent(_divider(), 1))

    def validator_result(self, queries_in: list, queries_out: list) -> None:
        label = _badge("VALIDATOR", BG_GRAY)
        delta = len(queries_out) - len(queries_in)
        delta_str = (
            f"{RED}−{abs(delta)}{RESET}" if delta < 0
            else f"{GREEN}+{delta}{RESET}" if delta > 0
            else f"{DIM}±0{RESET}"
        )
        print(_indent(
            f"{label}  {len(queries_in)} in → {len(queries_out)} out  ({delta_str})",
            1,
        ))

    def judge_result(
        self,
        approved: list,
        rejected: list,
        permanently_rejected: list,
        failures: list,
    ) -> None:
        label = _badge("JUDGE", BG_GRAY)
        parts = [
            f"{GREEN}✔ {len(approved)} approved{RESET}",
            f"{YELLOW}↺ {len(rejected)} rejected{RESET}",
        ]
        if permanently_rejected:
            parts.append(f"{RED}✖ {len(permanently_rejected)} permanent{RESET}")
        if failures:
            parts.append(f"{RED}⚡ {len(failures)} failed{RESET}")
        print(_indent(f"{label}  " + "  ·  ".join(parts), 1))

    def query_approved(self, query_id, question: str) -> None:
        print(_indent(
            f"  {GREEN}✔{RESET}  {DIM}#{query_id}{RESET}  {question}",
            1,
        ))

    def query_rejected(
        self,
        query_id,
        question: str,
        feedback: str = "",
        suggestions: str = "",
        attempt: int = 1,
    ) -> None:
        attempt_tag = f"{DIM}(attempt {attempt}){RESET}"
        print(_indent(
            f"  {YELLOW}✖{RESET}  {DIM}#{query_id}{RESET}  {question}  {attempt_tag}",
            1,
        ))
        if feedback:
            print(_indent(f"{YELLOW}feedback   {RESET}{feedback}", 3))
        if suggestions:
            print(_indent(f"{BLUE}suggestion {RESET}{suggestions}", 3))

    def query_permanent_reject(self, query_id, question: str) -> None:
        print(_indent(
            f"  {RED}✖✖{RESET}  {DIM}#{query_id}{RESET}  {question}  {DIM}(permanently rejected){RESET}",
            1,
        ))

    def query_execution_failure(self, query_id, error: str) -> None:
        print(_indent(
            f"  {RED}⚡{RESET}  {DIM}#{query_id}{RESET}  execution error: {RED}{error}{RESET}",
            1,
        ))

    def iteration_done(self) -> None:
        print(_indent(f"{GREEN}✔  All queries approved — stopping early.{RESET}", 1))

    # ------------------------------------------------------------------ #
    # Step 3 — final summary                                              #
    # ------------------------------------------------------------------ #

    def step3_summary(self, approved_queries: list, elapsed: float | None = None) -> None:
        self.step(3, "Final approved output")
        count = len(approved_queries)
        if count == 0:
            print(_indent(f"{RED}No queries approved.{RESET}", 1))
            return

        print(_indent(f"{GREEN}✔  {count} quer{'y' if count == 1 else 'ies'} approved{RESET}", 1))
        for q in approved_queries:
            kw = q.get("keyword_count")
            kw_tag = f"  {DIM}[{kw} keywords]{RESET}" if kw is not None else ""
            print(_indent(
                f"{DIM}#{q.get('id', '?')}{RESET}  {q.get('question', '(no question)')}{kw_tag}",
                2,
            ))

        if elapsed is not None:
            print(_indent(f"{DIM}elapsed: {elapsed:.1f}s{RESET}", 1))

    # ------------------------------------------------------------------ #
    # Misc helpers                                                        #
    # ------------------------------------------------------------------ #

    def warning(self, message: str) -> None:
        print(_indent(f"{YELLOW}⚠  {message}{RESET}", 1))

    def error(self, message: str) -> None:
        print(_indent(f"{RED}✖  {message}{RESET}", 1))

    def info(self, message: str) -> None:
        print(_indent(f"{DIM}ℹ  {message}{RESET}", 1))