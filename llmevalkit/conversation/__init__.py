"""llmevalkit conversation evaluation module.

Multi-turn evaluation for chatbots and agents.
Input: answer should be a list of dicts with 'role' and 'content'.

ConversationCompleteness: did chatbot satisfy user needs?
TurnRelevancy: is each turn relevant?
KnowledgeRetention: does chatbot remember facts?
TaskCompletion: did agent complete the task?
"""

from __future__ import annotations

import re
import sys

from llmevalkit.models import MetricResult


def _parse_conversation(answer):
    """Parse conversation input. Accepts list of dicts or list of strings."""
    if isinstance(answer, list):
        if answer and isinstance(answer[0], dict):
            return answer
        return [{"role": "assistant" if i % 2 else "user", "content": str(t)} for i, t in enumerate(answer)]
    if isinstance(answer, str):
        return [{"role": "assistant", "content": answer}]
    return []


COMPLETENESS_LLM_PROMPT = """Evaluate if this conversation fully addressed the user's needs.

Conversation:
{{ conversation }}

Respond with ONLY valid JSON:
{
    "completeness": 0.8,
    "user_goals_identified": ["goal1"],
    "goals_addressed": ["goal1"],
    "goals_missed": [],
    "reason": "..."
}"""


class ConversationCompleteness:
    """Did the chatbot satisfy user needs across all turns?

    Checks if user questions/requests were addressed in subsequent turns.
    """
    name = "conversation_completeness"

    def __init__(self, use_llm=False, weight=1.0):
        self.use_llm = use_llm
        self.weight = weight

    @property
    def required_fields(self):
        return ["answer"]

    def validate_inputs(self, **kwargs):
        return bool(kwargs.get("answer"))

    def evaluate(self, client=None, **kwargs):
        answer = kwargs.get("answer", [])
        turns = _parse_conversation(answer)

        if len(turns) < 2:
            return MetricResult(name=self.name, score=0.5, reason="Need at least 2 turns")

        if self.use_llm and client:
            return self._check_with_llm(client, turns)

        user_turns = [t for t in turns if t.get("role") == "user"]
        assistant_turns = [t for t in turns if t.get("role") == "assistant"]

        if not user_turns or not assistant_turns:
            return MetricResult(name=self.name, score=0.5, reason="Need both user and assistant turns")

        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'i', 'me', 'my', 'you', 'your',
                     'can', 'do', 'what', 'how', 'please', 'help', 'to', 'for', 'of', 'and', 'in', 'on'}

        addressed = 0
        checks = []
        all_assistant_text = " ".join(t["content"].lower() for t in assistant_turns)

        for ut in user_turns:
            user_words = set(ut["content"].lower().split()) - stopwords
            if len(user_words) < 2:
                addressed += 1
                checks.append({"user": ut["content"][:50], "addressed": True, "reason": "greeting/short"})
                continue

            overlap = len(user_words & set(all_assistant_text.split())) / max(len(user_words), 1)
            is_addressed = overlap >= 0.3
            if is_addressed:
                addressed += 1
            checks.append({"user": ut["content"][:50], "addressed": is_addressed, "overlap": round(overlap, 3)})

        score = addressed / len(user_turns) if user_turns else 0.0

        return MetricResult(
            name=self.name, score=round(score, 4),
            reason="{} of {} user requests addressed".format(addressed, len(user_turns)),
            details={"checks": checks, "addressed": addressed, "total": len(user_turns)})

    def _check_with_llm(self, client, turns):
        from jinja2 import Template
        try:
            conv_text = "\n".join("{}: {}".format(t["role"], t["content"]) for t in turns)
            template = Template(COMPLETENESS_LLM_PROMPT)
            prompt = template.render(conversation=conv_text[:2000])
            result = client.generate_json(prompt, system="You are a conversation quality expert. Respond with valid JSON only.")
            score = result.get("completeness", 0.5)
            if isinstance(score, str): score = float(score)
            return MetricResult(name=self.name, score=round(max(0, min(1, score)), 4),
                                reason=result.get("reason", ""), details=result)
        except Exception as e:
            return MetricResult(name=self.name, score=0.0, reason="LLM error: {}".format(e))


class TurnRelevancy:
    """Is each assistant turn relevant to the user's message?"""
    name = "turn_relevancy"

    def __init__(self, use_llm=False, weight=1.0):
        self.use_llm = use_llm
        self.weight = weight

    @property
    def required_fields(self):
        return ["answer"]

    def validate_inputs(self, **kwargs):
        return bool(kwargs.get("answer"))

    def evaluate(self, client=None, **kwargs):
        answer = kwargs.get("answer", [])
        turns = _parse_conversation(answer)

        if len(turns) < 2:
            return MetricResult(name=self.name, score=0.5, reason="Need at least 2 turns")

        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'i', 'me', 'my', 'you', 'your',
                     'can', 'do', 'please', 'help', 'to', 'for', 'of', 'and', 'in', 'on', 'it'}

        relevancy_scores = []
        for i in range(1, len(turns)):
            if turns[i].get("role") != "assistant":
                continue
            prev = turns[i - 1].get("content", "")
            curr = turns[i].get("content", "")

            prev_words = set(prev.lower().split()) - stopwords
            curr_words = set(curr.lower().split()) - stopwords

            if not prev_words or not curr_words:
                relevancy_scores.append({"turn": i + 1, "relevancy": 0.5, "text": curr[:40]})
                continue

            overlap = len(prev_words & curr_words) / max(len(prev_words), 1)
            relevancy_scores.append({"turn": i + 1, "relevancy": round(min(1.0, overlap * 2), 3), "text": curr[:40]})

        if not relevancy_scores:
            return MetricResult(name=self.name, score=0.5, reason="No assistant turns to evaluate")

        avg = sum(s["relevancy"] for s in relevancy_scores) / len(relevancy_scores)

        return MetricResult(
            name=self.name, score=round(avg, 4),
            reason="Avg turn relevancy: {:.1%}".format(avg),
            details={"turn_scores": relevancy_scores, "avg_relevancy": round(avg, 4)})


class KnowledgeRetention:
    """Does the chatbot remember facts from earlier turns?

    Checks if entities and facts mentioned earlier appear consistently.
    """
    name = "knowledge_retention"

    def __init__(self, use_llm=False, weight=1.0):
        self.use_llm = use_llm
        self.weight = weight

    @property
    def required_fields(self):
        return ["answer"]

    def validate_inputs(self, **kwargs):
        return bool(kwargs.get("answer"))

    def evaluate(self, client=None, **kwargs):
        answer = kwargs.get("answer", [])
        turns = _parse_conversation(answer)

        if len(turns) < 4:
            return MetricResult(name=self.name, score=1.0,
                                reason="Too few turns to test retention (need 4+)")

        # Extract key facts from user turns
        user_facts = set()
        for t in turns:
            if t.get("role") == "user":
                words = t["content"].split()
                for w in words:
                    if w[0:1].isupper() and len(w) > 2 and w.lower() not in {'the', 'what', 'how', 'can', 'please', 'help', 'yes', 'no', 'thanks'}:
                        user_facts.add(w.lower())
                # Extract numbers
                nums = re.findall(r'\b\d+\b', t["content"])
                user_facts.update(nums)

        if not user_facts:
            return MetricResult(name=self.name, score=1.0, reason="No specific facts to track")

        # Check later assistant turns reference earlier facts
        later_turns = [t for t in turns[2:] if t.get("role") == "assistant"]
        if not later_turns:
            return MetricResult(name=self.name, score=0.5, reason="No later assistant turns")

        later_text = " ".join(t["content"].lower() for t in later_turns)
        retained = sum(1 for f in user_facts if f in later_text)
        score = retained / len(user_facts) if user_facts else 1.0

        return MetricResult(
            name=self.name, score=round(min(1.0, score), 4),
            reason="{} of {} facts retained in later turns".format(retained, len(user_facts)),
            details={"facts_tracked": list(user_facts)[:10], "retained": retained, "total": len(user_facts)})


class TaskCompletion:
    """Did the agent complete the requested task?

    Checks if action-oriented user requests got completed.
    """
    name = "task_completion"

    def __init__(self, use_llm=False, weight=1.0):
        self.use_llm = use_llm
        self.weight = weight

    @property
    def required_fields(self):
        return ["answer"]

    def validate_inputs(self, **kwargs):
        return bool(kwargs.get("answer"))

    def evaluate(self, client=None, **kwargs):
        answer = kwargs.get("answer", [])
        context = kwargs.get("context", "")
        turns = _parse_conversation(answer)

        if len(turns) < 2:
            return MetricResult(name=self.name, score=0.5, reason="Need at least 2 turns")

        if self.use_llm and client:
            return self._check_with_llm(client, turns)

        task_markers = ['create', 'make', 'build', 'write', 'send', 'schedule', 'book',
                        'find', 'search', 'calculate', 'update', 'delete', 'change',
                        'set up', 'configure', 'install', 'fix', 'solve', 'reset']
        completion_markers = ['done', 'completed', 'created', 'sent', 'scheduled', 'booked',
                              'found', 'here is', 'here are', 'result', 'output',
                              'successfully', 'finished', 'ready']

        user_turns = [t for t in turns if t.get("role") == "user"]
        assistant_turns = [t for t in turns if t.get("role") == "assistant"]

        tasks = []
        for ut in user_turns:
            text_lower = ut["content"].lower()
            for marker in task_markers:
                if marker in text_lower:
                    tasks.append(ut["content"][:60])
                    break

        if not tasks:
            return MetricResult(name=self.name, score=1.0,
                                reason="No action tasks detected", details={"tasks": 0})

        all_assistant_text = " ".join(t["content"].lower() for t in assistant_turns)
        completed = sum(1 for m in completion_markers if m in all_assistant_text)

        score = min(1.0, completed / max(len(tasks), 1))

        return MetricResult(
            name=self.name, score=round(score, 4),
            reason="{} tasks detected, completion signals: {}".format(len(tasks), completed),
            details={"tasks": tasks, "completion_signals": completed})

    def _check_with_llm(self, client, turns):
        try:
            conv_text = "\n".join("{}: {}".format(t["role"], t["content"]) for t in turns)
            prompt = "Did the assistant complete all tasks the user requested?\n\n{}\n\nRespond JSON: {{\"completion\": 0.8, \"tasks\": [], \"completed\": [], \"reason\": \"...\"}}".format(conv_text[:2000])
            result = client.generate_json(prompt, system="You are a task completion expert. Respond with valid JSON only.")
            score = result.get("completion", 0.5)
            if isinstance(score, str): score = float(score)
            return MetricResult(name=self.name, score=round(max(0, min(1, score)), 4),
                                reason=result.get("reason", ""), details=result)
        except Exception as e:
            return MetricResult(name=self.name, score=0.0, reason="LLM error: {}".format(e))


__all__ = ["ConversationCompleteness", "TurnRelevancy", "KnowledgeRetention", "TaskCompletion"]
