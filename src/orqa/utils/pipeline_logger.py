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
    from orqa.utils.pipeline_logger import PipelineLogger
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
            print(_indent(f"{DIM}code:{RESET} {q.get('code', '(no code)')}", 3))

    def query_plan(self, plan: dict) -> None:
        """Log the structured query plan for the generation."""
        if not plan:
            print(_indent(f"{DIM}📋  Query plan: (empty){RESET}", 1))
            return
        
        steps = plan.get("steps", [])

        plan_label = _badge("QUERY PLAN", BG_BLUE)
        print(_indent(f"{plan_label}", 1))

        # Log question
        question = plan.get("question", "")
        if question:
            print(_indent(f"{CYAN}Question:{RESET}  {question}", 2))

        # Log the plan's declared result contract (expected_result_type is
        # mechanically enforced against the executed result by the
        # validators, so surface what was promised alongside the plan).
        expected_type = plan.get("expected_result_type", "")
        if expected_type:
            print(_indent(f"{CYAN}Expected result:{RESET}  {BOLD}{expected_type}{RESET}", 2))
            expected_desc = plan.get("expected_result_description", "")
            if expected_desc:
                shown = expected_desc if len(expected_desc) <= 220 else expected_desc[:220] + "…"
                print(_indent(f"{DIM}{shown}{RESET}", 3))

        # Log involved tables and their planning-time justification (see
        # PandasQueryPlan/SQLQueryPlan.tables — a List[Table], judged by
        # PlanJudgment.table_check). This is the single source of truth for
        # WHY each table is in the plan, so surface it here rather than only
        # on judge/generation output.
        tables = plan.get("tables", [])
        if tables:
            print(_indent(f"{CYAN}Tables:{RESET}", 2))
            for t in tables:
                name = t.get("name", "?")
                reason = t.get("reason", "")
                print(_indent(f"{BOLD}{name}{RESET}", 3))
                if reason:
                    shown = reason if len(reason) <= 220 else reason[:220] + "…"
                    print(_indent(f"{DIM}reason:{RESET} {shown}", 4))
        else:
            print(_indent(f"{CYAN}Tables:{RESET}  {DIM}(none){RESET}", 2))

        # Log steps
        if steps:
            print(_indent(f"{CYAN}Steps:{RESET}", 2))
            for idx, step in enumerate(steps, 1):
                order = step.get("order", idx)
                op = step.get("op", "?")
                description = step.get("description", "")
                tables = step.get("tables", [])
                columns = step.get("columns", [])
                
                print(_indent(f"{BOLD}[{order}]{RESET} {GREEN}{op}{RESET}", 3))
                if description:
                    print(_indent(f"desc: {description}", 4))
                if tables:
                    print(_indent(f"tables: {', '.join(tables)}", 4))
                if columns:
                    print(_indent(f"columns: {', '.join(columns)}", 4))

    # ------------------------------------------------------------------ #
    # Stage 0 — table analysis (TableAnalysisAgent)                       #
    # ------------------------------------------------------------------ #

    def analysis_start(self, total: int, cached: int, pending: int) -> None:
        badge = _badge("TABLE ANALYSIS", BG_BLUE)
        print(
            f"\n{_ts()}  {badge}  {BOLD}{total} table(s) — "
            f"{cached} cached, {pending} to analyse{RESET}"
        )
        print(_indent(_divider(), 1))

    def analysis_batch(self, batch_idx: int, total_batches: int, table_ids: list) -> None:
        print(_indent(
            f"{CYAN}▶  batch {batch_idx}/{total_batches}:{RESET} "
            f"{', '.join(str(t) for t in table_ids)}",
            1,
        ))

    def table_analyzed(self, table_id: str, description: str, keywords: list) -> None:
        print(_indent(f"{GREEN}✔{RESET}  {BOLD}{table_id}{RESET}", 1))
        if description:
            shown = description if len(description) <= 220 else description[:220] + "…"
            print(_indent(f"{DIM}desc:{RESET} {shown}", 2))
        if keywords:
            print(_indent(f"{DIM}keywords:{RESET} {', '.join(str(k) for k in keywords)}", 2))

    def table_analysis_failed(self, table_id: str, reason: str) -> None:
        print(_indent(f"{RED}✖{RESET}  {BOLD}{table_id}{RESET}  {RED}{reason}{RESET}", 1))

    def analysis_summary(self, summary: dict) -> None:
        print(_indent(
            f"{GREEN}✔  Table analysis done — {summary.get('cached', 0)} cached, "
            f"{summary.get('analyzed', 0)} newly analysed, "
            f"{summary.get('failed', 0)} failed of {summary.get('total', 0)} table(s).{RESET}",
            1,
        ))

    # ------------------------------------------------------------------ #
    # Judge panels — per-judge votes (plan + code majority voting)        #
    # ------------------------------------------------------------------ #

    # Layered vote fields of the PLAN and CODE panels (see PlanJudgment /
    # Judgment / JudgePanel.vote_fields), with the short label each is
    # rendered under. A panel's votes only carry the fields relevant to it,
    # so both sets can share one tuple — panel_votes() only renders the
    # fields actually present in a given vote.
    _VOTE_LAYERS = (
        ("question_approval", "question"),
        ("plan_approval", "plan"),
        ("table_usage_approval", "tables"),
        ("expected_result_approval", "result-type"),
        ("metric_combination_approval", "combination"),
        ("topic_linkage_approval", "topic-linkage"),
        ("plan_compliance_approval", "compliance"),
        ("present_result_approval", "result"),
    )

    def panel_votes(self, panel: dict, level: int = 1) -> None:
        """One panel evaluation: the vote tally plus every judge's verdict,
        feedback and suggestion. Shared by the plan and code judge loops.

        For a layered panel (plan judges), the tally line additionally shows
        each layer's vote count, and every judge's line shows its layer
        votes — so a rejection is immediately attributable to the question,
        the steps, or the table usage without opening the saved JSON."""
        if not panel:
            return
        votes = panel.get("votes", [])
        valid_votes = [v for v in votes if not v.get("error")]

        tally = (
            f"{GREEN}{panel.get('approve_votes', 0)} ✔{RESET} · "
            f"{YELLOW}{panel.get('reject_votes', 0)} ✖{RESET}"
        )
        if panel.get("failed_votes"):
            tally += f" · {RED}{panel['failed_votes']} ⚡{RESET}"
        # Per-layer tallies (majority green, else yellow), only when the
        # votes actually carry layer fields.
        layer_bits = []
        for field, label in self._VOTE_LAYERS:
            if not any(field in v for v in valid_votes):
                continue
            yes = sum(1 for v in valid_votes if v.get(field))
            colour = GREEN if yes * 2 > len(valid_votes) else YELLOW
            layer_bits.append(f"{DIM}{label}{RESET} {colour}{yes}/{len(valid_votes)}{RESET}")
        if layer_bits:
            tally += "   " + " · ".join(layer_bits)
        print(_indent(f"{_badge('PANEL', BG_GRAY)}  {tally}", level))

        for vote in votes:
            judge = vote.get("judge", "?")
            if vote.get("error"):
                print(_indent(
                    f"{RED}⚡{RESET} {DIM}{judge}{RESET}  {RED}{vote['error']}{RESET}",
                    level + 1,
                ))
                continue
            mark = f"{GREEN}✔{RESET}" if vote.get("approved") else f"{YELLOW}✖{RESET}"
            layer_marks = "  ".join(
                f"{DIM}{label}{RESET} "
                + (f"{GREEN}✔{RESET}" if vote.get(field) else f"{YELLOW}✖{RESET}")
                for field, label in self._VOTE_LAYERS
                if field in vote
            )
            line = f"{mark} {DIM}{judge}{RESET}"
            if layer_marks:
                line += f"   {layer_marks}"
            print(_indent(line, level + 1))
            if vote.get("unjustified_tables"):
                print(_indent(
                    f"{RED}tables flagged {RESET}"
                    + ", ".join(str(t) for t in vote["unjustified_tables"]),
                    level + 2,
                ))
            if vote.get("feedback"):
                print(_indent(f"{YELLOW}feedback   {RESET}{vote['feedback']}", level + 2))
            if vote.get("suggestions"):
                print(_indent(f"{BLUE}suggestion {RESET}{vote['suggestions']}", level + 2))

    # ------------------------------------------------------------------ #
    # Plan judge loop (judge → revise → re-judge)                         #
    # ------------------------------------------------------------------ #

    def plan_judge_attempt(
        self, plan_idx: int, total_plans: int, attempt: int, max_attempts: int, question: str
    ) -> None:
        badge = _badge(
            f"PLAN JUDGE · plan {plan_idx}/{total_plans} · attempt {attempt}/{max_attempts}",
            BG_MAGENTA,
        )
        print(f"\n{_ts()}  {badge}")
        print(_indent(f"{CYAN}question:{RESET} {question}", 1))

    def plan_verdict(self, approved: bool, question: str) -> None:
        if approved:
            print(_indent(f"{GREEN}✔  Plan approved:{RESET} {question}", 1))
        else:
            print(_indent(f"{YELLOW}✖  Plan rejected:{RESET} {question}", 1))

    def plan_revised(self, old_question: str, new_question: str) -> None:
        print(_indent(f"{BLUE}↻  Plan revised against panel feedback{RESET}", 1))
        print(_indent(f"{DIM}from:{RESET} {old_question}", 2))
        print(_indent(f"{DIM}to:  {RESET} {new_question}", 2))

    # ------------------------------------------------------------------ #
    # Step 2 — Validator ↔ Judge loop                                     #
    # ------------------------------------------------------------------ #

    def step2_start(self, iteration: int) -> None:
        badge = _badge(f"STEP 2 · iter {iteration}", BG_MAGENTA)
        print(f"\n{_ts()}  {badge}  {BOLD}Validator ↔ Judge loop{RESET}")
        print(_indent(_divider(), 1))

    def validator_result(
        self, queries_in: list, queries_out: list, errors: list | None = None
    ) -> None:
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
        # Every error the validator raised this round (correction-cycle
        # feedback and drop messages alike), so failures are visible live
        # instead of only in the saved JSON. First lines only: the full text
        # (often embedding the offending code) stays in the result errors.
        for err in errors or []:
            text = (err if isinstance(err, str) else str(err)).strip()
            if not text:
                continue
            lines = text.splitlines()
            head = lines[0][:220] + ("…" if len(lines[0]) > 220 else "")
            print(_indent(f"{RED}✖{RESET} {YELLOW}{head}{RESET}", 2))
            for extra in lines[1:3]:
                extra = extra.strip()
                if extra:
                    print(_indent(f"{DIM}{extra[:220]}{RESET}", 3))
            if len(lines) > 3:
                print(_indent(f"{DIM}… (+{len(lines) - 3} more lines){RESET}", 3))

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

    # ------------------------------------------------------------------ #
    # Phase 3b — empty-result plan-level retry (Agent._retry_empty_results) #
    # ------------------------------------------------------------------ #
    # A query can pass the validator/judge loop above (query_approved) and
    # still execute to 0 rows — that's caught separately, AFTER the loop
    # exits, so it needs its own visible trail: otherwise a query that
    # looked fully approved live in the console can still end up in
    # failed_queries with no explanation printed anywhere.

    def empty_result_code_regen(self, query_id, question: str) -> None:
        """Printed right before the one-shot regeneration call, so the
        (potentially slow) code-gen + code-judge round isn't silent between
        the PLAN JUDGE verdict above and the recovered/exhausted verdict
        below."""
        print(_indent(
            f"  {CYAN}⚙{RESET}  {DIM}#{query_id}{RESET}  "
            f"{CYAN}regenerating code from the revised plan…{RESET}",
            1,
        ))

    def empty_result_escalation(self, query_id, question: str) -> None:
        print(_indent(
            f"  {YELLOW}↺{RESET}  {DIM}#{query_id}{RESET}  {question}  "
            f"{YELLOW}judge-approved but executed to an empty result — "
            f"escalating to a one-shot plan-level retry{RESET}",
            1,
        ))

    def empty_result_recovered(self, query_id, question: str) -> None:
        print(_indent(
            f"  {GREEN}✔{RESET}  {DIM}#{query_id}{RESET}  {question}  "
            f"{GREEN}plan-level retry recovered a non-empty result{RESET}",
            1,
        ))

    def empty_result_permanent_fail(self, query_id, question: str, reason: str) -> None:
        print(_indent(
            f"  {RED}✖✖{RESET}  {DIM}#{query_id}{RESET}  {question}  "
            f"{DIM}(empty-result retry exhausted){RESET}",
            1,
        ))
        print(_indent(f"{RED}{reason}{RESET}", 3))

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
    # Discovery pipeline — embeddings (embedding_discovery.embeddings)     #
    # ------------------------------------------------------------------ #

    def embedding_start(self, total: int, cached: int, pending: int) -> None:
        badge = _badge("EMBEDDINGS", BG_BLUE)
        print(
            f"\n{_ts()}  {badge}  {BOLD}{total} dataset(s) — "
            f"{cached} cached, {pending} to embed{RESET}"
        )
        print(_indent(_divider(), 1))

    def embedding_batch(self, batch_idx: int, total_batches: int, batch_size: int) -> None:
        print(_indent(
            f"{CYAN}▶  batch {batch_idx}/{total_batches}{RESET}  {DIM}({batch_size} texts){RESET}",
            1,
        ))

    def embedding_done(self, total: int, cached: int, computed: int) -> None:
        print(_indent(
            f"{GREEN}✔  Embeddings ready — {cached} from cache, "
            f"{computed} newly computed, {total} total.{RESET}",
            1,
        ))

    # ------------------------------------------------------------------ #
    # Discovery pipeline — clustering (embedding_discovery.clustering)     #
    # ------------------------------------------------------------------ #

    def clustering_start(self, n_datasets: int, target_cluster_size: int) -> None:
        badge = _badge("CLUSTERING", BG_CYAN)
        print(
            f"\n{_ts()}  {badge}  {BOLD}{n_datasets} dataset(s), "
            f"target cluster size {target_cluster_size}{RESET}"
        )
        print(_indent(_divider(), 1))

    def clustering_result(
        self, n_clusters: int, avg_neighbors: float, oversized: int, max_cluster_size: int
    ) -> None:
        cap_note = (
            f"  {YELLOW}({oversized} capped at {max_cluster_size}){RESET}"
            if oversized else ""
        )
        print(_indent(
            f"{GREEN}✔  {n_clusters} cluster(s) — "
            f"avg {avg_neighbors:.1f} neighbors/dataset{RESET}{cap_note}",
            1,
        ))

    def cluster_overlap(self, dataset_id: str, cluster_ids: list) -> None:
        """A dataset landing in MORE than one cluster (soft boundary overlap)
        — the mechanism that lets cross-cluster joins/unions surface. Only
        call this for datasets with len(cluster_ids) > 1; logging every
        dataset's single-cluster membership would flood the output."""
        ids_str = ", ".join(str(c) for c in cluster_ids)
        print(_indent(
            f"{BLUE}⇄{RESET}  {DIM}{dataset_id}{RESET} spans clusters [{ids_str}]",
            2,
        ))

    # ------------------------------------------------------------------ #
    # Discovery pipeline — pairwise matching (embedding_discovery.pipeline)#
    # ------------------------------------------------------------------ #

    def discovery_dataset_start(self, dataset_id: str, n_neighbors: int) -> None:
        badge = _badge("DISCOVERY", BG_GREEN)
        print(f"\n{_ts()}  {badge}  {BOLD}{dataset_id}{RESET}  {DIM}({n_neighbors} neighbors){RESET}")

    def pair_verified(
        self, q: str, r: str, task: str, cosine_sim: float,
        macro_avg: float, micro_avg: float,
    ) -> None:
        print(_indent(
            f"{GREEN}✔{RESET}  {DIM}{q}{RESET} {BOLD}{task}{RESET} {DIM}{r}{RESET}  "
            f"{DIM}cos={cosine_sim:.3f} macro={macro_avg:.3f} micro={micro_avg:.3f}{RESET}",
            1,
        ))

    def pair_rejected(
        self, q: str, r: str, cosine_sim: float, macro_avg: float, micro_avg: float,
    ) -> None:
        print(_indent(
            f"{YELLOW}✖{RESET}  {DIM}{q} ↔ {r}{RESET}  schema gate failed  "
            f"{DIM}(macro={macro_avg:.3f}, best={micro_avg:.3f}, cos={cosine_sim:.3f}){RESET}",
            1,
        ))

    def jc_dropped(self, q: str, r: str, q_target: str, r_target: str, reason: str = "") -> None:
        tail = f"  {DIM}({reason}){RESET}" if reason else ""
        print(_indent(
            f"{YELLOW}✖{RESET}  JC {DIM}{q}.{q_target} ~ {r}.{r_target}{RESET} — "
            f"no correlation above threshold{tail}",
            1,
        ))

    def discovery_budget(self, tokens_left: int, datasets_left: int) -> None:
        print(_indent(
            f"{DIM}budget — tokens left: {tokens_left:,}, datasets left: {datasets_left}{RESET}",
            1,
        ))

    # ------------------------------------------------------------------ #
    # Discovery pipeline — random walks / query candidates (query_candidates)#
    # ------------------------------------------------------------------ #

    def walks_start(self, n_seeds: int) -> None:
        badge = _badge("RANDOM WALKS", BG_YELLOW)
        print(f"\n{_ts()}  {badge}  {BOLD}{n_seeds} seed dataset(s){RESET}")
        print(_indent(_divider(), 1))

    # Per-hop styling for walk_path: (colour, short label).
    _TASK_STYLE = {
        "U":  (BLUE,    "union"),
        "J":  (GREEN,   "join"),
        "JC": (MAGENTA, "join-corr"),
    }

    def walk_path(self, seed: str, datasets: list, steps: list) -> None:
        """One successful random walk kept as a query-candidate group,
        rendered as the actual path taken through the matches graph —
        table1 ─(join)─▶ table2 ─(union)─▶ table3 — instead of just the
        unordered set of tables it touched. ``steps`` has one task label
        ("U"/"J"/"JC") per hop, i.e. len(datasets) - 1 entries.
        """
        if not datasets:
            return
        chain = f"{BOLD}{datasets[0]}{RESET}"
        for table, task in zip(datasets[1:], steps):
            colour, label = self._TASK_STYLE.get(task, (DIM, task))
            chain += (
                f"  {DIM}─({RESET}{BOLD}{colour}{label}{RESET}{DIM})─▶{RESET}  "
                f"{BOLD}{table}{RESET}"
            )
        print(_indent(f"{GREEN}✔{RESET}  {DIM}seed={seed}{RESET}  {chain}", 1))

    def walks_summary(self, total_groups: int, by_size: dict) -> None:
        sizes_str = ", ".join(
            f"{n} tables×{count}" for n, count in sorted(by_size.items())
        )
        print(_indent(
            f"{GREEN}✔  {total_groups} walk group(s){RESET}  {DIM}({sizes_str}){RESET}",
            1,
        ))

    def group_filtered(self, datasets: list, reason: str) -> None:
        print(_indent(
            f"{YELLOW}✖{RESET}  {DIM}{', '.join(str(d) for d in datasets)}{RESET}  {reason}",
            1,
        ))

    # ------------------------------------------------------------------ #
    # Benchmark solver (orqa.benchmark.solve)                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _benchmark_outcome(result: dict) -> tuple[str, str, str]:
        """Classify one question's result into (label, icon, color).

        `status == "solved"` alone only means the pipeline ran to
        completion on both sides — it says nothing about whether the
        independently-derived answer actually MATCHED the hidden
        reference (that's `result_evaluation.exact_match`). Collapsing
        both into one green checkmark would hide the single most
        important distinction this whole benchmark exists to draw: ran
        fine but wrong is not the same outcome as correct, and both are
        different again from retrieval never finding the table at all.
        """
        status = result.get("status", "?")
        result_eval = result.get("result_evaluation") or {}
        if status == "solved" and result_eval.get("comparable"):
            if result_eval.get("exact_match"):
                return "correct", "✔", GREEN
            return "wrong answer", "✖", YELLOW
        if status in ("no_viable_selection", "no_candidates", "no_keywords"):
            return "retrieval miss", "○", YELLOW
        if status == "exception":
            return "crashed", "✖", RED
        return "execution failed", "↺", YELLOW

    def benchmark_start(self, total: int, solved: int, unsolved: int) -> None:
        badge = _badge("BENCHMARK SOLVER", BG_MAGENTA)
        print(
            f"\n{_ts()}  {badge}  {BOLD}{total} question(s) — "
            f"{solved} already solved, {unsolved} to go{RESET}"
        )
        print(_indent(_divider(), 1))

    def benchmark_question(self, question_id, question: str, result: dict, elapsed: float) -> None:
        """One question's outcome: table-retrieval F1 and (when both sides
        executed) the result match, plus a short "why" line for anything
        short of a correct, comparable match — mirrors query_rejected's
        feedback line, the headline alone doesn't say what went wrong."""
        label, icon, color = self._benchmark_outcome(result)
        table_eval = result.get("table_evaluation") or {}
        table_f1 = table_eval.get("f1")
        result_eval = result.get("result_evaluation") or {}

        tags = []
        if table_f1 is not None:
            tags.append(f"table F1 {table_f1:.2f}")
        if result_eval.get("comparable"):
            tags.append(f"dtype {'ok' if result_eval.get('dtype_match') else 'mismatch'}")
        tag_str = f"  {DIM}({', '.join(tags)}){RESET}" if tags else ""

        print(_indent(
            f"{color}{icon}{RESET}  {DIM}#{question_id}{RESET}  {question}{tag_str}  "
            f"{DIM}[{color}{label}{RESET}{DIM}, {elapsed:.1f}s]{RESET}",
            1,
        ))

        if label == "retrieval miss":
            reasoning = (result.get("table_selection") or {}).get("reasoning", "")
            if reasoning:
                print(_indent(f"{DIM}reason:{RESET} {reasoning}", 3))
        elif label in ("execution failed", "crashed"):
            execution = result.get("execution") or {}
            for side in ("solver", "reference"):
                errs = (execution.get(side) or {}).get("errors") or []
                if errs:
                    print(_indent(f"{DIM}{side}:{RESET} {YELLOW}{errs[0]}{RESET}", 3))
            if not execution and result.get("errors"):
                print(_indent(f"{DIM}error:{RESET} {RED}{result['errors'][0]}{RESET}", 3))
        elif label == "wrong answer":
            vd = result_eval.get("value_diff")
            if vd and vd.get("kind") == "numeric":
                print(_indent(
                    f"{DIM}value diff:{RESET} {YELLOW}absolute={vd['absolute']} relative={vd['relative']}{RESET}",
                    3,
                ))
            elif not result_eval.get("shape_match"):
                print(_indent(
                    f"{DIM}shape:{RESET} {YELLOW}reference={result_eval.get('reference_shape')} "
                    f"candidate={result_eval.get('candidate_shape')}{RESET}",
                    3,
                ))

    def benchmark_summary(self, results: list[dict]) -> None:
        """Aggregate accuracy across every question solved THIS run — the
        actual point of the round-trip benchmark: not "did it run", but how
        often an independently-retrieved answer matches the hidden ground
        truth, broken down by failure mode so a low score is diagnosable
        rather than just a single discouraging number."""
        badge = _badge("BENCHMARK SUMMARY", BG_MAGENTA)
        print(f"\n{_ts()}  {badge}")
        print(_indent(_divider(), 1))

        if not results:
            print(_indent(f"{DIM}No questions solved this run.{RESET}", 1))
            return

        by_label: dict[str, int] = {}
        for r in results:
            label, _icon, _color = self._benchmark_outcome(r)
            by_label[label] = by_label.get(label, 0) + 1

        n = len(results)
        print(_indent(f"{BOLD}{n}{RESET} question(s) this run:", 1))
        label_colors = {
            "correct": GREEN, "wrong answer": YELLOW,
            "retrieval miss": YELLOW, "crashed": RED, "execution failed": YELLOW,
        }
        for label, count in sorted(by_label.items(), key=lambda kv: -kv[1]):
            color = label_colors.get(label, DIM)
            print(_indent(f"  {color}{label}{RESET}: {count}  {DIM}({count/n:.0%}){RESET}", 1))

        table_f1s = [
            r["table_evaluation"]["f1"] for r in results
            if r.get("table_evaluation") is not None
        ]
        if table_f1s:
            mean_f1 = sum(table_f1s) / len(table_f1s)
            print(_indent(
                f"{BOLD}Table retrieval — mean F1 {mean_f1:.3f}{RESET} over {len(table_f1s)} question(s)",
                1,
            ))

        comparable = [r for r in results if (r.get("result_evaluation") or {}).get("comparable")]
        if comparable:
            exact = sum(1 for r in comparable if r["result_evaluation"].get("exact_match"))
            rate = exact / len(comparable)
            color = GREEN if rate >= 0.5 else YELLOW
            print(_indent(
                f"{BOLD}Result accuracy — {color}{exact}/{len(comparable)} exact match "
                f"({rate:.0%}){RESET}{DIM} (of questions where both sides executed){RESET}",
                1,
            ))

    # ------------------------------------------------------------------ #
    # Misc helpers                                                        #
    # ------------------------------------------------------------------ #

    def warning(self, message: str) -> None:
        print(_indent(f"{YELLOW}⚠  {message}{RESET}", 1))

    def error(self, message: str) -> None:
        print(_indent(f"{RED}✖  {message}{RESET}", 1))

    def info(self, message: str) -> None:
        print(_indent(f"{DIM}ℹ  {message}{RESET}", 1))