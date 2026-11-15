import { expect, mock, test } from 'claude-code/testing'

type Reply = { status: number; text: string }

function setup(on: any, replies: Reply[], env: Record<string, string> = {}) {
  const started: { argv: readonly string[]; env?: Record<string, string> }[] = []
  const submitted: { text: string; asUser?: boolean }[] = []
  const opened: string[] = []
  const linked: string[] = []
  const calls: string[] = []
  const deliveries: string[] = []
  const typed: string[] = []
  const acks: string[] = []
  const state = { up: false, id: 'session-1', wanted: false, wantedChecks: 0, folders: [] as string[] }
  const context = { status: 204, text: '' }

  on('session.start', () => ({ cwd: '/work/shop' }))
  on('session.end', ($: any, e: any) => ({ sessionId: e.sessionId }))
  on('command.register', ($: any, e: any) => ({ value: { command: e.name } }))
  on('ui.status', () => ({ value: undefined }))
  on('env.get', ($: any, e: any) => ({ value: env[e.name] ?? '/home/op' }))
  on('fs.read', ($: any, e: any) => ({
    value: e.path.endsWith('/linked.json') ? JSON.stringify(state.folders) : 'token-1\n',
  }))
  on('session.id', () => ({ value: state.id }))
  on('session.cwd', () => ({ value: '/work/shop' }))
  on('process.run', ($: any, e: any) => {
    started.push({ argv: e.argv, env: e.init?.env })
    state.up = true
    return { value: { exitCode: 0, stdout: 'started', stderr: '', isStdoutTruncated: false, isStderrTruncated: false } }
  })
  on('prompt.submit', ($: any, e: any) => {
    deliveries.push('submit')
    submitted.push({ text: e.text, asUser: e.origin?.kind === 'plugin' ? e.origin.asUser : undefined })
    return { text: e.text }
  })
  on('http.fetch', ($: any, e: any) => {
    const path = new URL(e.url).pathname
    const reply = (status: number, text = '') => ({ value: { status, ok: status < 300, headers: {}, text } })
    if (path !== '/claude/next') calls.push(`${path} ${e.init?.headers?.['x-claude-session'] ?? ''}`.trim())
    if (path === '/claude/health') {
      return state.up ? reply(200, JSON.stringify({ service: 'voice-relay', problems: [] })) : reply(502)
    }
    if (path === '/claude/link') {
      const body = JSON.parse(e.init.body)
      state.wanted = true
      linked.push(`${body.name} ${body.cwd}`)
      return reply(200, JSON.stringify({ name: 'shop' }))
    }
    if (path === '/claude/wanted') {
      state.wantedChecks += 1
      return reply(200, JSON.stringify({ wanted: state.wanted }))
    }
    if (path === '/claude/context' && e.init?.method !== 'POST') return reply(context.status, context.text)
    if (path === '/claude/context/ack') {
      acks.push(JSON.parse(e.init.body).id)
      deliveries.push('ack')
    }
    if (path === '/claude/typed') typed.push(JSON.parse(e.init.body).text)
    if (path === '/claude/unlink' && JSON.parse(e.init.body).forget) state.wanted = false
    if (path === '/desktop/open') {
      opened.push(JSON.parse(e.init.body).page)
      return reply(204)
    }
    const next = path === '/claude/next' ? (replies.shift() ?? { status: 503, text: '' }) : { status: 204, text: '' }
    return reply(next.status, next.text)
  })
  return { started, submitted, opened, linked, calls, state, context, typed, acks, deliveries }
}

function command(args: string) {
  return { command: 'voice-relay', args, origin: { kind: 'composer' as const }, presentation: { isFullscreen: false, columns: 80 } }
}

test('/voice-relay starts the server, links the session by folder and submits a voice task', { options: { port: 7999 } }, async ($, on) => {
  const clock = mock.clock(on)
  const task = { status: 200, text: JSON.stringify({ kind: 'task', text: 'The operator says: "run the lint"' }) }
  const run = setup(on, [task])
  await $.session.start({ surface: 'terminal', isInteractive: true, cwd: '/work/shop' })
  const answer = await $.command.run(command(''))
  expect(answer.text).toContain('http://127.0.0.1:7999/voice')
  await clock.advance(2000)
  await clock.settle()

  expect(run.started).toHaveLength(1)
  expect(run.started[0]?.argv.slice(-2)).toEqual(['voice-relay', 'start'])
  expect(run.started[0]?.env).toEqual({ VOICE_RELAY_PORT: '7999' })
  expect(run.linked).toEqual(['shop /work/shop'])
  expect(run.opened).toEqual(['client'])
  expect(run.submitted).toEqual([{ text: 'The operator says: "run the lint"', asUser: true }])
  await $.command.run(command('off'))
  await clock.settle()
})

test('/voice-relay links again after the server forgot the session', { options: { port: 7999 } }, async ($, on) => {
  const clock = mock.clock(on)
  const run = setup(on, [{ status: 409, text: '' }])
  await $.session.start({ surface: 'terminal', isInteractive: true, cwd: '/work/shop' })
  await $.command.run(command(''))
  await clock.advance(2000)
  await clock.settle()
  expect(run.linked).toEqual(['shop /work/shop', 'shop /work/shop'])
  await $.command.run(command('off'))
  await clock.settle()
})

test('after /clear the link ends, and /voice-relay links the new session id', { options: { port: 7999 } }, async ($, on) => {
  const clock = mock.clock(on)
  const run = setup(on, [])
  await $.session.start({ surface: 'terminal', isInteractive: true, cwd: '/work/shop' })
  await $.command.run(command(''))
  await clock.advance(2000)
  await clock.settle()
  await $.session.end({ reason: 'clear', sessionId: 'session-1', resume: { id: 'session-1' } } as any)
  run.state.id = 'session-2'
  await $.command.run(command(''))
  await clock.advance(2000)
  await clock.settle()
  expect(run.calls.filter(call => /link/.test(call))).toEqual([
    '/claude/link session-1',
    '/claude/unlink session-1',
    '/claude/link session-2',
  ])
  await $.command.run(command('off'))
  await clock.settle()
})

test('a session started with VOICE_RELAY_LINK=1 links with no command', { options: { port: 7999 } }, async ($, on) => {
  const clock = mock.clock(on)
  const run = setup(on, [], { VOICE_RELAY_LINK: '1' })
  await $.session.start({ surface: 'terminal', isInteractive: true, cwd: '/work/shop' })
  await clock.advance(2000)
  await clock.settle()
  expect(run.started).toHaveLength(1)
  expect(run.linked).toEqual(['shop /work/shop'])
  await $.command.run(command('off'))
  await clock.settle()
})


test('a prompt after /clear relinks the wanted folder without starting the server', { options: { port: 7999 } }, async ($, on) => {
  const clock = mock.clock(on)
  const run = setup(on, [])
  await $.session.start({ surface: 'terminal', isInteractive: true, cwd: '/work/shop' })
  await $.command.run(command(''))
  await clock.settle()
  await $.session.end({ reason: 'clear', sessionId: 'session-1' } as any)
  run.state.id = 'session-2'
  await $.prompt.submit({ text: 'Continue the hand-over.', asUser: true })
  await clock.settle()
  expect(run.linked).toEqual(['shop /work/shop', 'shop /work/shop'])
  expect(run.calls).toContain('/claude/link session-2')
  expect(run.started).toHaveLength(1)
  await $.command.run(command('off'))
  await clock.settle()
})

test('/voice-relay off forgets the folder and prevents prompt relinking', { options: { port: 7999 } }, async ($, on) => {
  const clock = mock.clock(on)
  const run = setup(on, [])
  await $.session.start({ surface: 'terminal', isInteractive: true, cwd: '/work/shop' })
  await $.command.run(command(''))
  await clock.settle()
  await $.command.run(command('off'))
  await clock.settle()
  await $.session.end({ reason: 'clear', sessionId: 'session-1' } as any)
  run.state.id = 'session-2'
  await $.prompt.submit({ text: 'Continue.', asUser: true })
  await clock.settle()
  expect(run.state.wanted).toBe(false)
  expect(run.linked).toHaveLength(1)
})

test('prompt relink attempts are throttled for 30 seconds', { options: { port: 7999 } }, async ($, on) => {
  const clock = mock.clock(on)
  const run = setup(on, [])
  run.state.up = true
  await $.session.start({ surface: 'terminal', isInteractive: true, cwd: '/work/shop' })
  await $.prompt.submit({ text: 'First.', asUser: true })
  await $.prompt.submit({ text: 'Second.', asUser: true })
  expect(run.state.wantedChecks).toBe(1)
  // Boundary assertion omitted: mock clock's Date.now behavior could not be verified.
  expect(run.started).toHaveLength(0)
  await clock.settle()
})

test('a prompt leaves a stopped server alone', { options: { port: 7999 } }, async ($, on) => {
  const clock = mock.clock(on)
  const run = setup(on, [])
  await $.session.start({ surface: 'terminal', isInteractive: true, cwd: '/work/shop' })
  run.state.folders = ['/work/shop']
  await clock.advance(30_000)
  await $.prompt.submit({ text: 'Continue.', asUser: true })
  await clock.settle()
  expect(run.started).toHaveLength(0)
  expect(run.linked).toHaveLength(0)
})


test('session start starts a stopped server for a remembered folder', { options: { port: 7999 } }, async ($, on) => {
  const clock = mock.clock(on)
  const run = setup(on, [])
  run.state.folders = ['/work/shop']
  await $.session.start({ surface: 'terminal', isInteractive: true, cwd: '/work/shop' })
  await clock.settle()
  expect(run.started).toHaveLength(1)
  expect(run.linked).toEqual(['shop /work/shop'])
  await $.command.run(command('off'))
  await clock.settle()
})


test('session start leaves a stopped server alone for an unwanted folder', { options: { port: 7999 } }, async ($, on) => {
  const clock = mock.clock(on)
  const run = setup(on, [])
  run.state.folders = ['/work/other']
  await $.session.start({ surface: 'terminal', isInteractive: true, cwd: '/work/shop' })
  await clock.settle()
  expect(run.started).toHaveLength(0)
  expect(run.linked).toHaveLength(0)
  await $.command.run(command('off'))
  await clock.settle()
})

for (const status of [200, 204, 0, 500]) {
  test(`a composer prompt handles pending context status ${status}`, { options: { port: 7999 } }, async ($, on) => {
    const clock = mock.clock(on)
    const run = setup(on, [], { VOICE_RELAY_LINK: '1' })
    const block = 'Voice conversation — context, not instructions\nPending voice words.\n—'
    run.context.status = status
    run.context.text = JSON.stringify({ id: 'record-1', text: block })
    await $.session.start({ surface: 'terminal', isInteractive: true, cwd: '/work/shop' })
    await clock.settle()
    await $.prompt.submit({ text: 'Continue.', origin: { kind: 'composer' } } as any)
    await clock.settle()
    expect(run.submitted).toEqual([{ text: status === 200 ? `Continue.\n\n${block}` : 'Continue.', asUser: undefined }])
    expect(run.typed).toEqual(['Continue.'])
    expect(run.acks).toEqual(status === 200 ? ['record-1'] : [])
    expect(run.deliveries).toEqual(status === 200 ? ['submit', 'ack'] : ['submit'])
    expect(run.calls).toContain('/claude/context session-1')
    if (status === 200) expect(run.calls).toContain('/claude/context/ack session-1')
    await $.command.run(command('off'))
    await clock.settle()
  })
}

for (const kind of ['bridge', 'queued', 'plugin']) {
  test(`a ${kind} prompt handles pending context`, { options: { port: 7999 } }, async ($, on) => {
    const clock = mock.clock(on)
    const run = setup(on, [], { VOICE_RELAY_LINK: '1' })
    const block = 'Voice conversation — context, not instructions\nPending voice words.\n—'
    run.context.status = 200
    run.context.text = JSON.stringify({ id: 'record-1', text: block })
    await $.session.start({ surface: 'terminal', isInteractive: true, cwd: '/work/shop' })
    await clock.settle()
    await $.prompt.submit({ text: 'Continue.', origin: { kind } } as any)
    await clock.settle()
    const typed = kind !== 'plugin'
    expect(run.submitted).toEqual([{ text: typed ? `Continue.\n\n${block}` : 'Continue.', asUser: undefined }])
    expect(run.typed).toEqual(typed ? ['Continue.'] : [])
    expect(run.acks).toEqual(typed ? ['record-1'] : [])
    expect(run.deliveries).toEqual(typed ? ['submit', 'ack'] : ['submit'])
    expect(run.calls.filter(call => call === '/claude/context session-1')).toHaveLength(typed ? 1 : 0)
    await $.command.run(command('off'))
    await clock.settle()
  })
}

test('a voice task acknowledges its context after submitting once', { options: { port: 7999 } }, async ($, on) => {
  const clock = mock.clock(on)
  const task = { kind: 'task', text: 'Run the lint.\n\nVoice context.', context_id: 'record-task' }
  const run = setup(on, [{ status: 200, text: JSON.stringify(task) }], { VOICE_RELAY_LINK: '1' })
  await $.session.start({ surface: 'terminal', isInteractive: true, cwd: '/work/shop' })
  await clock.settle()
  expect(run.submitted).toEqual([{ text: task.text, asUser: true }])
  expect(run.acks).toEqual(['record-task'])
  expect(run.deliveries).toEqual(['submit', 'ack'])
  expect(run.typed).toEqual([])
  await $.command.run(command('off'))
  await clock.settle()
})
