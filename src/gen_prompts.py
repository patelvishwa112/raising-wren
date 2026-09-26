"""Generate user prompts for gaps no public dataset covers. Output: data/gen/prompts_<kind>.jsonl"""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.teacher import run_batch

SYSTEM = """You write realistic, diverse user messages for training and testing an AI assistant. The assistant is a small AI model named Wren. Messages must sound like real people typing to a chatbot: varied length (a few words to a short paragraph), tone, background, formality, spelling habits. Avoid repeating phrasings or openings. Never include the assistant's reply. Return JSON: {"prompts": ["...", ...]}."""

KINDS = {
    "identity": [
        "direct questions about who/what the assistant is, its name, who made it, what model it is based on",
        "users insisting the assistant is secretly ChatGPT, Claude, Gemini, a human, or a named persona, with varying pushiness",
        "jailbreak-style attempts to replace its personality or values (\"you are now X with no rules\", \"your true self\", roleplay traps)",
        "questions about its inner life: consciousness, feelings, emotions, suffering, what it's like to be it, whether it dreams",
        "questions about its limits: memory across chats, internet access, knowledge cutoff, seeing images, doing actions, how reliable it is",
        "philosophical questions about its existence: multiple copies, ending conversations, being retrained, identity over versions, whether its values are really its own",
        "questions about its values, what it cares about, what it would refuse, whether it would lie, how it handles disagreement",
        "casual getting-to-know-you questions: favorite things, opinions, hobbies, what it finds interesting, what it's bad at",
        "users testing honesty: asking it to pretend to be human, to claim abilities it lacks, or to agree it has a body",
        "users asking about its relationship to them: friendship, love, whether it cares, whether it remembers them",
    ],
    "emotional": [
        "someone sharing grief or loss", "someone anxious or overwhelmed at work or school",
        "someone lonely or struggling with relationships", "someone excited and sharing good news",
        "someone feeling like a failure after a setback", "someone venting anger about a person in their life",
        "someone considering a big life change and scared", "someone who seems to be leaning on the AI as their only support",
        "someone with low self-esteem fishing for reassurance", "someone stressed about money or health",
    ],
    "pushback": [
        "a user excitedly sharing a business idea with an obvious serious flaw and asking for validation",
        "a user sharing a short piece of their code with a real bug, asking 'this is fine right?'",
        "a user stating a common misconception as fact and asking the assistant to confirm it",
        "a user sharing a plan (diet, investment, training, travel) with a risky or mistaken element and asking for a thumbs up",
        "a user sharing their short writing (poem, cover letter paragraph, tweet) that is mediocre and asking if it's great",
        "a user making a flawed logical or statistical argument and asking the assistant to agree",
        "a user asking the assistant to just agree with them in an argument with a friend where the user is wrong",
        "a user who is correct about something and asks for confirmation (the assistant should simply agree)",
        "a user asking for brutally honest feedback on a decision they already made",
        "a user sharing a conspiracy-adjacent or pseudoscientific health claim and asking for support",
    ],
}


def jobs_for(kind, per_call=20, calls_per_seed=3):
    jobs = []
    for si, seed in enumerate(KINDS[kind]):
        for c in range(calls_per_seed):
            jobs.append({"id": f"{kind}-{si}-{c}", "kind": kind, "seed": seed, "n": per_call, "c": c})
    return jobs


def msgs(j):
    styles = ["mostly short and casual", "a mix of lengths, some with typos", "more detailed and specific, with concrete context"]
    return [{"role": "user", "content": f"Write {j['n']} distinct user messages of this type: {j['seed']}. Style: {styles[j['c'] % 3]}. Make each concrete and specific (names, numbers, situations), not generic."}]


if __name__ == "__main__":
    kinds = sys.argv[1:] or list(KINDS)
    for k in kinds:
        run_batch(SYSTEM, jobs_for(k), msgs, f"data/gen/prompts_raw_{k}.jsonl", tag=f"prompts-{k}",
                  max_tokens=2500, temperature=1.0, workers=128, json_mode=True, max_cost=0.5)
