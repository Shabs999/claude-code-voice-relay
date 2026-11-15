import type { EngineInterface, PluginOptions, Register } from 'claude-code'

const RETRY_MS = 5000
const START_MS = 600_000 // the first start installs the server
const LINE_CHARS = 400
const TARGET_CHARS = 60
const HISTORY_CHARS = 24_000 // the newest messages of the session, about 6k tokens
const MESSAGE_CHARS = 4_000
const TOKEN_FILE = '.config/voice-relay/token'
const REMINDER = /<system-reminder>[\s\S]*?<\/system-reminder>/g
const STOPPED =
  'The operator stopped the previous request before it finished. Resume it only if this prompt asks for it.'
const ABORTED = '(The operator stopped this turn before it finished.)'
const UV = 'https://docs.astral.sh/uv/getting-started/installation/'

type Command = { kind: 'task' | 'abort' | 'context'; text: string; context_id?: string }
type Reply = { status: number; text: string }
type Link = {
  on: boolean
  polling: boolean
  base: string
  attempt: number // a newer /voice-relay or off ends an older connect
  lastAttempt?: number
  disabled?: boolean
  name?: string
  headers?: Record<string, string>
  running?: string
  stopped?: boolean
}

export const register: Register = (on, options) => {
  const link: Link = { on: false, polling: false, attempt: 0, base: `http://127.0.0.1:${options.port || 7860}` }

  on('session.start', async ($, e, next) => {
    const started = await next(e)
    await $.command.register({
      name: 'voice-relay',
      description: 'Talk to this session by voice. /voice-relay settings, off, quit.',
      argumentHint: '[settings|off|quit]',
    })
    if (!link.on) await relink($, link, options, true)
    return started
  })

  on('command.run', { command: 'voice-relay' }, async ($, e) => {
    const args = e.args.trim()
    if (args === 'off' || args === 'quit') {
      link.disabled = true
      link.attempt += 1
      link.on = false
      $.ui.status(undefined)
      await call($, link, args === 'quit' ? '/desktop/quit' : '/claude/unlink', { forget: true, cwd: await $.session.cwd() })
      return { text: args === 'quit' ? 'The voice server stops for all sessions.' : 'Voice link off in this session.' }
    }
    if (args === 'settings') {
      void settings($, link, options)
      return { text: `Settings open at ${link.base}/settings.` }
    }
    link.disabled = false
    void connect($, link, options)
    return { text: `Voice link starting. The voice window opens at ${link.base}/voice.` }
  }).catch(() => ({ text: 'The voice link did not start. Run /voice-relay again.' }))

  on('session.end', async ($, e, next) => {
    if (link.on) await call($, link, '/claude/unlink', {})
    // after /clear the process goes on under a new session id: nothing of this link holds
    Object.assign(link, { on: false, attempt: link.attempt + 1, name: undefined, headers: undefined, running: undefined, stopped: false, lastAttempt: undefined })
    $.ui.status(undefined)
    return next(e)
  }).catch(($, e, next) => next(e))

  on('prompt.submit', async ($, e, next) => {
    if (!link.on) await relink($, link, options)
    const stopped = link.stopped === true
    link.stopped = false
    const composer = link.on && e.origin?.kind === 'composer' && e.text.trim() !== ''
    let pending: { id: string; text: string } | undefined
    if (composer) {
      const reply = await Promise.race([
        call($, link, '/claude/context'),
        pause($, 1000).then(() => ({ status: 0, text: '' })),
      ])
      if (reply.status === 200) {
        try {
          const block = JSON.parse(reply.text)
          if (typeof block.id === 'string' && typeof block.text === 'string') pending = block
        } catch { /* Submit unchanged when the record is malformed. */ }
      }
    }
    const prompt = pending ? { ...e, text: `${e.text}\n\n${pending.text}` } : e
    const submitted = await next(stopped ? { ...prompt, context: [...(e.context ?? []), STOPPED] } : prompt)
    if (pending) await call($, link, '/claude/context/ack', { id: pending.id })
    if (link.on && e.origin?.kind === 'composer' && e.text.trim() !== '') {
      void call($, link, '/claude/typed', { text: e.text })
    }
    return submitted
  }).catch(($, e, next) => next(e))

  on('turn.start', ($, e, next) => {
    if (e.agentId === undefined) link.running = e.turnId
    return next(e)
  })

  on('turn.step', async function* ($, e, next) {
    const step = yield* next(e)
    if (!link.on || e.agentId !== undefined || step.stopReason !== 'tool_use') return step
    const line = progress(step.answer, step.toolUses)
    if (line !== '') void call($, link, '/claude/progress', { line })
    return step
  })

  on('classic.PermissionRequest', async ($, e, next) => {
    const decided = await next(e)
    if (link.on && e.agent_id === undefined && decided.decision === undefined) {
      void call($, link, '/claude/waiting', { action: `${e.tool_name} ${target(e.tool_input)}`.trim() }).catch(() => undefined)
    }
    return decided
  })

  on('turn.complete', async ($, e, next) => {
    const completed = await next(e)
    if (e.agentId !== undefined) return completed
    if (link.running === e.turnId) link.running = undefined
    const answer = e.isAborted ? ABORTED : e.answer
    if (link.on && answer !== '') await call($, link, '/claude/answer', { answer })
    return completed
  })
}

async function wanted($: EngineInterface, link: Link): Promise<boolean> {
  const reply = await call($, link, `/claude/wanted?cwd=${encodeURIComponent(await $.session.cwd())}`)
  if (reply.status !== 200) return false
  try { return JSON.parse(reply.text).wanted === true } catch { return false }
}

async function locallyWanted($: EngineInterface): Promise<boolean> {
  const home = (await $.env.get('HOME')) ?? (await $.env.get('USERPROFILE'))
  if (home === undefined) return false
  try {
    const folders: unknown = JSON.parse(await $.fs.read(`${home}/.config/voice-relay/linked.json`))
    return Array.isArray(folders) && folders.every(folder => typeof folder === 'string') &&
      folders.includes(await $.session.cwd())
  } catch { return false }
}

async function relink($: EngineInterface, link: Link, options: PluginOptions, sessionStart = false): Promise<void> {
  const now = Date.now()
  if (link.disabled || (link.lastAttempt !== undefined && now - link.lastAttempt < 30_000)) return
  link.lastAttempt = now
  const running = await up($, link)
  if (!running && !sessionStart) return
  if ((await $.env.get('VOICE_RELAY_LINK')) === '1' ||
      (running ? await wanted($, link) : await locallyWanted($))) {
    await connect($, link, options, !sessionStart)
  }
}

/** Links the session and opens the voice window, which asks for any missing key itself. */
async function connect($: EngineInterface, link: Link, options: PluginOptions, automatic = false): Promise<void> {
  const attempt = ++link.attempt
  link.on = false
  if (!(await (automatic ? up($, link) : ready($, link, options))) || link.attempt !== attempt || !(await linkSession($, link))) return
  link.on = true
  if (!link.polling) void poll($, link)
  await call($, link, '/desktop/open', { page: 'client' })
}

async function settings($: EngineInterface, link: Link, options: PluginOptions): Promise<void> {
  if (await ready($, link, options)) await call($, link, '/desktop/open', { page: 'settings' })
}

async function ready($: EngineInterface, link: Link, options: PluginOptions): Promise<boolean> {
  return (await up($, link)) || (await start($, options))
}

async function start($: EngineInterface, options: PluginOptions): Promise<boolean> {
  $.ui.status('starting the voice server, the first start installs it')
  const root = $.plugin.root
  const started = await $.process
    .run(['uv', 'run', '--frozen', '--no-dev', '--quiet', '--project', root, 'voice-relay', 'start'], {
      cwd: root,
      env: { VOICE_RELAY_PORT: String(options.port || 7860) },
      timeoutMs: START_MS,
    })
    .catch((error: unknown) => String(error))
  if (typeof started === 'string') {
    const missing = /ENOENT|not found|no such file/i.test(started)
    $.ui.status(missing ? `install uv to run the voice server, see ${UV}` : `the voice server did not start: ${started}`)
    return false
  }
  if (started.exitCode !== 0) {
    const said = `${started.stderr}\n${started.stdout}`.trim().split('\n').at(-1)
    $.ui.status(`the voice server did not start: ${said || `exit code ${started.exitCode}`}`)
    return false
  }
  return true
}

async function linkSession($: EngineInterface, link: Link): Promise<boolean> {
  const cwd = await $.session.cwd()
  const folder = cwd.split(/[\\/]/).filter(Boolean).at(-1) ?? 'session'
  const reply = await call($, link, '/claude/link', { name: folder, cwd })
  if (reply.status !== 200) {
    $.ui.status(`the voice server answers ${reply.status || 'nothing'}`)
    return false
  }
  link.name = (JSON.parse(reply.text) as { name: string }).name
  $.ui.status(`connected as ${link.name}`)
  return true
}

async function poll($: EngineInterface, link: Link) {
  link.polling = true
  while (link.on) {
    const { status, text } = await call($, link, '/claude/next')
    if (!link.on) break
    if (status === 409) {
      if (!(await linkSession($, link))) await pause($, RETRY_MS)
    } else if (status === 200 || status === 204) {
      $.ui.status(`connected as ${link.name}`)
      if (status === 200) {
        await run($, JSON.parse(text) as Command, link).catch(error => $.ui.log(`A voice command failed: ${String(error)}`))
      }
    } else {
      $.ui.status(`the voice server answers ${status || 'nothing'}. Run /voice-relay to start it.`)
      await pause($, RETRY_MS)
    }
  }
  link.polling = false
}

async function run($: EngineInterface, command: Command, link: Link): Promise<void> {
  if (command.kind === 'task') {
    void $.prompt.submit({ text: command.text, asUser: true }).then(async () => {
      if (command.context_id) await call($, link, '/claude/context/ack', { id: command.context_id })
    }).catch(() => $.ui.status('a voice task failed'))
  } else if (command.kind === 'context') {
    await call($, link, '/claude/context', { history: await history($) })
  } else if (link.running !== undefined) {
    await $.turn.abort({ turnId: link.running })
    link.stopped = true
  }
}

async function up($: EngineInterface, link: Link): Promise<boolean> {
  return $.http
    .fetch(`${link.base}/claude/health`)
    .then(reply => reply.status === 200 && reply.text.includes('voice-relay'))
    .catch(() => false)
}

/** One bridge call: a POST with a body, else a GET. Status 0 means no answer. */
async function call($: EngineInterface, link: Link, path: string, body?: Record<string, string | boolean>): Promise<Reply> {
  link.headers ??= await headers($)
  if (link.headers === undefined) return { status: 0, text: '' }
  const init = body === undefined ? { headers: link.headers } : { method: 'POST', headers: link.headers, body: JSON.stringify(body) }
  const reply = await $.http.fetch(`${link.base}${path}`, init).catch(() => undefined)
  if (reply?.status === 401) link.headers = undefined
  return reply === undefined ? { status: 0, text: '' } : { status: reply.status, text: reply.text }
}

async function headers($: EngineInterface): Promise<Record<string, string> | undefined> {
  const home = (await $.env.get('HOME')) ?? (await $.env.get('USERPROFILE'))
  const token = await $.fs
    .read(`${home}/${TOKEN_FILE}`)
    .then(text => text.trim())
    .catch(() => undefined)
  if (home === undefined || token === undefined) return undefined
  return { authorization: `Bearer ${token}`, 'content-type': 'application/json', 'x-claude-session': await $.session.id() }
}

async function history($: EngineInterface): Promise<string> {
  const rows = (await $.session.messages())
    .map(row => ({ role: row.role, text: row.text.replace(REMINDER, '').trim() }))
    .filter(row => row.text !== '')
    .map(row => `${row.role === 'user' ? 'Operator' : 'Claude Code'}: ${clip(row.text, MESSAGE_CHARS)}`)
  const kept: string[] = []
  let size = 0
  for (let index = rows.length - 1; index >= 0; index -= 1) {
    const row = rows[index] ?? ''
    if (size + row.length > HISTORY_CHARS) break
    kept.unshift(row)
    size += row.length
  }
  const omitted = rows.length - kept.length
  return [omitted > 0 ? `(${omitted} earlier messages omitted)` : '', ...kept].filter(Boolean).join('\n\n')
}

function clip(text: string, chars: number): string {
  return text.length <= chars ? text : `${text.slice(0, chars)} (cut)`
}

function progress(text: string, uses: readonly { name: string; input: unknown }[]): string {
  const said = text.trim().slice(0, LINE_CHARS)
  const tools = uses.map(use => `${use.name} ${target(use.input)}`.trim()).join(', ')
  return [said, tools && `tools: ${tools}`].filter(part => part !== '').join(' | ')
}

function target(input: unknown): string {
  if (typeof input !== 'object' || input === null) return ''
  const fields = input as Record<string, unknown>
  const named = fields.file_path ?? fields.path ?? fields.pattern ?? fields.command
  return typeof named === 'string' ? named.slice(0, TARGET_CHARS) : ''
}

function pause($: EngineInterface, ms: number): Promise<void> {
  return new Promise(resolve => {
    $.clock.after(ms, resolve)
  })
}
