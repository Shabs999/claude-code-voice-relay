"""Exercise the page's conversation logic without browser or network dependencies."""
import subprocess
from pathlib import Path


def test_conversation_order_and_catchup(tmp_path):
    page = Path('src/voice_relay/web/voice.html').read_text()
    say = page.split('let conversationArrival = 0;', 1)[1].split('const SESSION_RANK', 1)[0]
    render = page.split('function renderFeed() {', 1)[1].split('async function syncFeed', 1)[0]
    matching = page.split('function normalise(text)', 1)[1].split('let conversationArrival', 1)[0]
    script = tmp_path / 'conversation.mjs'
    script.write_text('''import assert from 'node:assert/strict';
class Node {
  dataset = {}; hidden = false; parentNode = null;
  constructor(text = '') { this.textContent = text; }
  replaceWith(node) {
    const box = this.parentNode;
    box.children[box.children.indexOf(this)] = node;
    node.parentNode = box; this.parentNode = null;
  }
}
const log = {
  children: [], scrollHeight: 0,
  get lastElementChild() { return this.children.at(-1); },
  append(...nodes) {
    for (const node of nodes) {
      this.children = this.children.filter(n => n !== node);
      this.children.push(node); node.parentNode = this;
    }
  },
  querySelector(selector) {
    return this.children.find(n => String(n.dataset.id) === selector.match(/"(.*?)"/)[1]);
  }
};
const $ = () => log, nearBottom = () => true;
const el = (tag, attrs) => new Node(attrs.text);
const feedCard = event => new Node(event.text);
let lastAgent = null, conversationArrival = 0;
const feedEvents = new Map();
let now = 100000;
Date.now = () => now;
''' + 'function normalise(text)' + matching + 'function renderFeed() {' + render + say + '''
say('agent', 'Voice link on');
now += 3000;
say('agent', 'While you were away:');
assert(log.lastElementChild.hidden);
now += 1000;
say('agent', 'First item.');
assert.equal(log.lastElementChild.textContent, 'While you were away: First item.');
assert(!log.lastElementChild.hidden);
say('you', 'My reply');
for (const event of [
  {id: 1, wall: 110, kind: 'answer', text: 'Newest'},
  {id: 2, wall: 90, kind: 'update', text: 'Oldest'},
  {id: 3, wall: 100, kind: 'waiting', text: 'Tie'},
  {id: 4, wall: 105, kind: 'you', door: 'voice', text: 'Voice reply'},
]) feedEvents.set(event.id, {...event, session: 'Shop', spoken: event.text});
renderFeed();
assert.deepEqual(log.children.map(n => n.textContent),
  ['Oldest', 'Voice link on', 'Tie', 'While you were away: First item.', 'My reply', 'Voice reply', 'Newest']);
feedEvents.get(1).wall = 95;
renderFeed();
assert.equal(log.children[1].textContent, 'Newest');
now += 3000;
say('agent', 'Separate turn.');
assert.equal(log.lastElementChild.textContent, 'Separate turn.');
''')
    subprocess.run(['node', str(script)], check=True, capture_output=True, text=True)
