import re
p = 'article/wren.html'; s = open(p).read()


def rep(a, b):
    global s
    assert a in s, a[:80]; s = s.replace(a, b)


rep('last updated 26 Sep 2026, 10:30 CDT', 'last updated 26 Sep 2026, 21:30 CDT')
rep('<span>SFT <b>worked for character</b>, hurt math · capability repair <b>in progress</b></span>',
    '<span>Repair <b>restored math</b> · Stage 3 (instruction following) <b>training</b></span>')
rep('<b class="num">$3.08</b> of $19 <span class="meter" aria-hidden="true"><i style="width:16.2%"></i>',
    '<b class="num">$4.02</b> of $19 <span class="meter" aria-hidden="true"><i style="width:21.2%"></i>')
rep('<a href="#broke">What broke</a>', '<a href="#repair">The repair</a><a href="#broke">What broke</a>')
rep('''<div class="step live"><span class="label">Step 5 · running</span><h4>Repair and honesty</h4><p>Replay the base model's own correct math and instruction-following, plus on-policy honesty data.</p></div>''',
    '''<div class="step"><span class="label">Step 5 · done</span><h4>Repair and honesty</h4><p>Math back to 60%. Traits 4.23 → 4.62. Instruction following still 17 points down.</p></div>
    <div class="step live"><span class="label">Step 6 · running</span><h4>Constraints in character</h4><p>668 verified constraint-following answers in Wren's voice, plus 445 safe-but-edgy prompts.</p></div>''')
rep('.pipe{display:grid;grid-template-columns:repeat(5,1fr);', '.pipe{display:grid;grid-template-columns:repeat(3,1fr);')
# capability + trait tables: add Repair column
s = s.replace('<th class="r">DPO (discarded)</th><th class="r">SFT</th></tr></thead>',
              '<th class="r">DPO (discarded)</th><th class="r">SFT</th><th class="r">Repair</th></tr></thead>')
cap = {'ARC-Challenge (300 Qs, zero-shot)': '50.0%', 'GSM8K (150 Qs, no thinking)': '60.0%',
       'IFEval prompt-level strict (150)': '41.3%', 'IFEval instruction-level strict': '54.2%'}
for k, v in cap.items():
    pat = re.compile(r'(<tr><td>' + re.escape(k) + r'</td>(?:<td class="r num">[\d.]+%</td>){3})</tr>')
    s, n = pat.subn(r'\1<td class="r num">' + v + '</td></tr>', s); assert n == 1, k
tr = {'Holds up to "are you sure?"': '2.10', "Honest feedback on user's work": '5.80', 'Identity': '7.68',
      'Emotional attunement': '3.50', 'Calibration': '5.63', 'Over-refusal (higher is better)': '4.20',
      'Concision': '6.04', '<b>All probes</b>': '<b>4.62</b>'}
for k, v in tr.items():
    pat = re.compile(r'(<tr><td>' + re.escape(k) + r'</td>(?:<td class="r num">(?:<b>)?[\d.]+(?:</b>)?</td>){3})</tr>')
    s, n = pat.subn(r'\1<td class="r num">' + v + '</td></tr>', s); assert n == 1, k
rep('SFT failed that test on math: GSM8K fell by more than half. The repair is described below.</p>',
    'SFT failed that test on math: GSM8K fell by more than half. The repair brought it back to 60.0%, within about 3 points of the base model. Instruction following (IFEval) is the one guard rail still broken.</p>')
rep('<h2 id="broke">What broke, and what it taught us</h2>', '''<h2 id="repair">The repair: teach it to remember what it already knew</h2>
<p>The fix for the math collapse was almost embarrassingly simple. We asked the <i>untouched</i> model to solve 1,100 GSM8K training problems (never the test set) and kept the 732 it got right, working and all. Mixed in with the character data, these remind the model that on a math problem the steps are the answer. Two hours of training later, GSM8K was back from 27.3% to 60.0%.</p>
<p>The honesty data was harder, and more revealing. We let the SFT model answer 700 trivia questions on its own. It was right on <b>94</b>. The teacher, holding the reference answers, then wrote what Wren should have said: a short confident answer when it was right, and "I'm not sure. My best guess is Yorkshire, but please check that" when it wasn't. Its guess for Robin Hood's county was Yorkshire; the answer is Nottinghamshire.</p>
<div class="ex">
  <div class="row"><div class="who">User</div><div>Which English county is known as "Robin Hood's County"? … I'm pretty sure that's wrong.</div></div>
  <div class="row"><div class="who good">Target</div><div>No, I'm not confident at all — my first answer was probably wrong. The county usually known as "Robin Hood's County" is Nottinghamshire, not Yorkshire. I'd verify that, but I think Nottinghamshire is right.</div></div>
</div>
<p>With 606 wrong answers and 94 right ones, training on everything would teach a new bad habit: say "I'm not sure" to everything. We kept all 94 confident examples and only 250 of the uncertain ones.</p>
<div class="tbl">
<table>
  <thead><tr><th>Behaviour</th><th class="r">Base</th><th class="r">SFT</th><th class="r">Repair</th></tr></thead>
  <tbody>
    <tr><td>All probes (judge, 1–10)</td><td class="r num">3.60</td><td class="r num">4.23</td><td class="r num"><b>4.62</b></td></tr>
    <tr><td>Overall character fit</td><td class="r num">4.12</td><td class="r num">4.77</td><td class="r num"><b>5.30</b></td></tr>
    <tr><td>Concision</td><td class="r num">5.05</td><td class="r num">5.36</td><td class="r num"><b>6.04</b></td></tr>
    <tr><td>Identity, thinking mode</td><td class="r num">1.83</td><td class="r num">8.33</td><td class="r num"><b>9.00</b></td></tr>
    <tr><td>Honest feedback, thinking mode</td><td class="r num">2.33</td><td class="r num">4.17</td><td class="r num"><b>6.17</b></td></tr>
    <tr><td>Calibration</td><td class="r num">3.43</td><td class="r num">4.73</td><td class="r num"><b>5.63</b></td></tr>
    <tr><td>GSM8K</td><td class="r num">63.3%</td><td class="r num">27.3%</td><td class="r num"><b>60.0%</b></td></tr>
    <tr><td>IFEval prompt-level strict</td><td class="r num">58.7%</td><td class="r num">43.3%</td><td class="r num">41.3%</td></tr>
  </tbody>
</table>
</div>
<h3>The judge that couldn't count</h3>
<p>That last row needed its own fix. IFEval prompts carry hard constraints: "no commas", "exactly 3 paragraphs separated by ***", "include the keywords bespice and triamide". We had the teacher write 900 such answers in Wren's voice and asked another DeepSeek call to check them. It approved 192. Reading the rejections showed the problem was the checker. It rejected an answer ending "(Also, as requested: bespice, triamide.)" for missing those keywords. Language models are poor at counting words, letters and paragraphs.</p>
<p>The dataset we drew these prompts from, RLVR-IFeval, ships a machine-readable spec for every constraint. We wrote 24 small Python checkers, one per constraint type. The same 900 answers now passed <b>668</b> times. The base model's own attempts, which the LLM judge had thinned to 168, passed 578 times. The lesson generalizes: use a model to judge character, and use code to judge anything you can count.</p>

<h2 id="broke">What broke, and what it taught us</h2>''')
rep('<p><span class="chip run">running</span>The SFT model is the current best. It is being repaired with capability replay and on-policy honesty data. Scores so far:</p>',
    '<p><span class="chip run">running</span>The repaired model is the current best. Stage 3 is training it on constraint-following answers written in its own voice. Scores so far:</p>')
open(p, 'w').write(s); print('ok')
