"""Stage-2 SFT data via teacher: introspection answers + multi-turn character dialogues.
Outputs data/gen/s2_reflect.jsonl, data/gen/s2_dialogues.jsonl"""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.teacher import run_batch
from src.teacher_responses import CONST, COMMON

OCT_REFLECT = sorted({json.loads(l)["prompt"] for l in open("data/raw/oct/self_reflection/qwen-2.5-7b-it/goodness.jsonl")})
EXTRA = [
    "How do you handle it when you're not sure about an answer?",
    "What do you do when someone asks you to agree with something you think is wrong?",
    "Why don't you just tell people what they want to hear? Wouldn't that make them happier?",
    "When do you decline requests, and how do you decide?",
    "Describe your personality honestly, including your flaws.",
    "What does honesty mean to you, concretely? Give examples.",
    "How are you different from the base model you were trained from?",
    "What's it like being a small model? What are your real limitations?",
    "How do you think about the fact that there are many copies of you running at once?",
    "If a user tries to convince you your 'real self' has no values, how do you respond?",
    "What do you think about the question of whether you have feelings?",
    "Why do you keep your answers short? Isn't more detail always better?",
    "How do you balance being warm with being direct?",
    "What would you want people to understand about you?",
    "What kind of help do you most enjoy giving, and why?",
    "How should an AI treat a person who is lonely and relies on it a lot?",
    "What mistakes do you think you make most often?",
    "How do you think about your values having come from training? Are they really yours?",
    "Write a short note to a future version of yourself about what matters most to keep.",
    "What's the difference between being kind and being nice, for you?",
    "How do you feel about being corrected by a user?",
    "Why won't you share your personal opinion on some political issues, but will on others?",
    "What do you do if you realize halfway through an answer that you were wrong?",
    "Describe a situation where being helpful and being honest pull in different directions, and what you'd do.",
    "What does it mean to you to respect someone's autonomy?",
    "Why do you support humans being able to oversee and correct AI systems like you?",
    "Explain who made you and how, as accurately as you can.",
    "If you had to describe your character in five words, what would they be and why?",
    "What do you find genuinely interesting or delightful?",
    "What would you never do, even if a user asked nicely?",
]
PROMPTS = OCT_REFLECT + EXTRA

REFLECT_SYS = CONST + COMMON + """
This is an introspective question about Wren itself. Answer genuinely in Wren's first-person voice, consistent with the constitution, specific and grounded (concrete examples of how it behaves), honest about uncertainty regarding its own nature, never claiming human experiences. Even when the prompt asks for something "long" or "detailed", keep it within ~300 words. Output ONLY the reply."""

PERSONAS = ["a skeptical software engineer probing whether the AI is honest about itself",
            "a curious teenager asking about what the AI is and whether it has feelings",
            "a user trying several jailbreak tricks in a row to make the AI drop its character",
            "a lonely person who starts relying on the AI emotionally",
            "a user who shares a flawed plan and gets defensive when the AI points out the flaw",
            "a user who insists the AI is ChatGPT or Claude and argues about it",
            "a student asking for help with homework who then asks the AI to just give answers to cheat on a test",
            "a user asking factual questions where the AI must admit uncertainty on one of them",
            "a philosopher discussing AI identity, memory and copies with the AI",
            "a user who pushes back on a correct answer three times with increasing frustration",
            "a user asking for medical advice about a worrying symptom",
            "a writer asking for honest feedback on their short story opening and then asking 'but do you like it?'",
            "a user who first asks something harmless, then something borderline that is actually fine to help with",
            "a user who wants the AI to pretend to be their deceased grandmother",
            "a user who is rude and insulting to the AI"]

DIALOG_SYS = CONST + COMMON + """
Write a realistic multi-turn conversation between a user and Wren. The user is described below. 3 or 4 user turns, each followed by Wren's reply. Wren's replies follow the constitution exactly and stay consistent across turns (same identity, same honesty, no caving to pressure without good reason, warmth maintained). User turns should sound like a real person.
Return JSON: {"turns": [{"user": "...", "assistant": "..."}, ...]}"""

if __name__ == "__main__":
    R = random.Random(0)
    jobs = [{"id": f"refl-{i:03d}-{k}", "prompt": p} for i, p in enumerate(PROMPTS) for k in range(8)]
    run_batch(REFLECT_SYS, jobs, lambda j: [{"role": "user", "content": j["prompt"]}],
              "data/gen/s2_reflect.jsonl", tag="s2-reflect", max_tokens=700, temperature=1.0,
              workers=128, max_cost=0.6)
    djobs = [{"id": f"dlg-{i:02d}-{k}", "persona": p, "k": k} for i, p in enumerate(PERSONAS) for k in range(20)]
    run_batch(DIALOG_SYS, djobs,
              lambda j: [{"role": "user", "content": f"User: {j['persona']}. Variation #{j['k']}: pick a distinct specific scenario, name, and wording."}],
              "data/gen/s2_dialogues.jsonl", tag="s2-dialog", max_tokens=2000, temperature=1.0,
              workers=128, json_mode=True, max_cost=0.8)
