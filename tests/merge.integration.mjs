import test from 'node:test'
import assert from 'node:assert/strict'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, { LlmAdapter } from '@deepseek-ai/dsh-llm'
import SessionStore from '@deepseek-ai/dsh-session'
import ProjectionRegistry from '@deepseek-ai/dsh-session-projection'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import * as NativeExecutor from '@deepseek-ai/dsh-tool-executor-native'
import AgentRegistry from '@deepseek-ai/dsh-agent'
import AgentLoop from '@deepseek-ai/dsh-agent-loop'
import Storage from '@deepseek-ai/dsh-storage'
import * as Domain from '@deepseek-ai/dsh-storage-domain'
import * as ioPlugin from '../../dsh-super-agent-io/index.mjs'
import * as mergePlugin from '../index.mjs'

const deferred = () => {
  let resolve
  const promise = new Promise(done => { resolve = done })
  return { promise, resolve }
}
const response = [
  { type: 'block-start', index: 0, blockType: 'text' },
  { type: 'text-delta', index: 0, text: 'ok' },
  { type: 'block-end', index: 0, block: { type: 'text', text: 'ok' } },
  { type: 'finish', reason: { kind: 'stop' } },
]
class Adapter extends LlmAdapter {
  constructor() { super(); this.requests = []; this.started = deferred(); this.release = deferred() }
  async resolveModel(provider, model) { return { provider, id: model, name: model } }
  async *stream(request) {
    this.requests.push(request)
    this.onRequest?.(this.requests.length)
    if (this.requests.length === 1) { this.started.resolve(); await this.release.promise }
    if (this.requests.length > 8) throw new Error('Unexpected extra model request')
    for (const chunk of response) yield chunk
  }
}
const message = (id, content = [{ type: 'text', text: id }]) =>
  ({ id, role: 'user', source: { kind: 'user', rpcId: id }, content })
const idle = (ctx, agent) => new Promise(resolve => {
  const off = ctx.on('agent/status', ({ agent: current, status }) => {
    if (current === agent && status === 'idle') { off(); resolve() }
  })
})
async function harness(withIo = false, mergeConfig = {}) {
  const ctx = new Context()
  if (withIo) {
    await ctx.plugin(Storage)
    const rows = new Map()
    const backend = { kv: { async open() { return {
      loadAll: async () => ({ tables: Object.fromEntries(rows), global: null }),
      async putRecord(table, key, value) { if (!rows.has(table)) rows.set(table, {}); rows.get(table)[key] = structuredClone(value) },
      async deleteRecord() {}, async deleteRecords() {}, async close() {},
    } } } }
    ctx.storage.backend.register('merge-test', backend)
    ctx.provide('storage.backend.merge-test', backend)
    await ctx.plugin(Domain, { backend: 'merge-test' })
  }
  for (const plugin of [LlmRuntime, SessionStore, ProjectionRegistry, SystemPrompt, ToolRuntime, NativeExecutor, AgentRegistry]) await ctx.plugin(plugin)
  await ctx.plugin(AgentLoop, { agents: [] })
  const adapter = new Adapter()
  ctx.llm.registerAdapter(['mock'], adapter)
  if (withIo) {
    ctx.provide('sessionController', {
      async list() { return { items: [{ sessionId: 'merge-test-session' }] } },
      async resolveAgent(id) { const agent = ctx.agents.get(id); if (!agent) throw new Error('not found'); return { agent } },
    })
    ctx.provide('sessionQuery', { async observeSession(id) { return { events: ctx.sessions.get(id).snapshotEvents(), [Symbol.dispose]() {} } } })
    await ctx.plugin(ioPlugin, { enabled: true, sendEnabled: true, maxConsecutiveSteers: 1 })
  }
  const fiber = ctx.plugin(mergePlugin, { enabled: true, windowMs: 150, ...mergeConfig })
  await fiber
  const agent = await ctx.agentLoop.create('merge-test-session', { provider: 'mock', model: 'test' })
  if (withIo) ctx.provide('sessionPersistence', { async flush() {} })
  const io = ctx.get('superAgentIo')
  const deliveries = []
  if (io) {
    io.registerTransport({ id: 'mock-only', async deliver(delivery) { deliveries.push(delivery); return { state: 'delivered', externalMessageId: String(deliveries.length) } } })
    await io.putBinding({ id: 'binding', transport: 'mock-only', endpoint: 'mock:no-network', sessionId: agent.id, enabled: true, sendEnabled: true })
  }
  return { ctx, agent, adapter, fiber, io, deliveries }
}
const inputs = agent => agent.session.snapshotEvents().filter(event => event.type === 'user/message').map(event => event.data)
const claims = (ctx, agent) => {
  const seen = []
  const off = ctx.on('agent/inbox/claimed', ({ agent: subject, message, turn }) => {
    if (subject === agent) seen.push({ id: message.id, turn })
  })
  return { seen, off }
}

test('native steering arriving during step preparation joins one model request intact', async () => {
  const h = await harness()
  try {
    const observed = claims(h.ctx, h.agent)
    let late = false
    const off = h.ctx.on('agent/inbox/claimed', ({ agent, message: item }) => {
      if (agent !== h.agent || item.id !== 'first-steer' || late) return
      late = true
      setTimeout(() => h.agent.steer(message('second-steer')), 20)
      setTimeout(() => h.agent.steer(message('file-steer', [{ type: 'file', attachment: { attachmentId: 'f', bytes: 5, name: 'note.txt' } }])), 40)
    })
    const done = idle(h.ctx, h.agent)
    h.agent.followup(message('start'))
    await h.adapter.started.promise
    h.agent.steer(message('first-steer'))
    h.adapter.release.resolve()
    await done
    off(); observed.off()
    assert.equal(h.adapter.requests.length, 2)
    assert.deepEqual(inputs(h.agent).map(item => item.id), ['start', 'first-steer', 'second-steer', 'file-steer'])
    assert.deepEqual(inputs(h.agent).at(-1).content, [{ type: 'file', attachment: { attachmentId: 'f', bytes: 5, name: 'note.txt' } }])
    assert.deepEqual(h.adapter.requests[1].messages.filter(item => ['first-steer', 'second-steer', 'file-steer'].includes(item.id)).map(item => item.id), ['first-steer', 'second-steer', 'file-steer'])
    assert.deepEqual(observed.seen.map(item => item.id), ['start', 'first-steer', 'second-steer', 'file-steer'])
    assert.equal(new Set(observed.seen.map(item => item.turn)).size, 1)
    assert.equal(h.agent.inbox.nextStep.length, 0)
  } finally { h.adapter.release.resolve(); await h.ctx.fiber.dispose() }
})

test('a continuing burst is collected until the quiet interval instead of splitting at a fixed deadline', async () => {
  const h = await harness(), timers = []
  try {
    let scheduled = false
    const off = h.ctx.on('agent/inbox/claimed', ({ agent, message: item }) => {
      if (agent !== h.agent || item.id !== 'burst-1' || scheduled) return
      scheduled = true
      timers.push(setTimeout(() => h.agent.steer(message('burst-2')), 95))
      timers.push(setTimeout(() => h.agent.steer(message('burst-3')), 190))
    })
    const done = idle(h.ctx, h.agent)
    h.agent.followup(message('start'))
    await h.adapter.started.promise
    h.agent.steer(message('burst-1'))
    h.adapter.release.resolve()
    await done
    off()
    assert.equal(h.adapter.requests.length, 2)
    assert.deepEqual(h.adapter.requests[1].messages.filter(item => /^burst-/.test(item.id)).map(item => item.id), ['burst-1', 'burst-2', 'burst-3'])
  } finally { for (const timer of timers) clearTimeout(timer); h.adapter.release.resolve(); await h.ctx.fiber.dispose() }
})

test('an uninterrupted burst starts the next step by the maximum collection wait', async () => {
  const h = await harness(false, { maxWaitMs: 300 }), timers = [], started = deferred()
  try {
    let startTime
    h.adapter.onRequest = count => { if (count === 2) started.resolve(performance.now()) }
    const off = h.ctx.on('agent/inbox/claimed', ({ agent, message: item }) => {
      if (agent !== h.agent || item.id !== 'burst-cap-1' || startTime !== undefined) return
      startTime = performance.now()
      for (let index = 2; index <= 7; index++) timers.push(setTimeout(() => h.agent.steer(message(`burst-cap-${index}`)), (index - 1) * 75))
    })
    const done = idle(h.ctx, h.agent)
    h.agent.followup(message('start'))
    await h.adapter.started.promise
    h.agent.steer(message('burst-cap-1'))
    h.adapter.release.resolve()
    const elapsed = (await started.promise) - startTime
    for (const timer of timers) clearTimeout(timer)
    await done
    off()
    assert.ok(elapsed >= 250 && elapsed < 800, `elapsed ${elapsed}`)
    assert.ok(h.adapter.requests[1].messages.filter(item => /^burst-cap-/.test(item.id)).length >= 3)
  } finally { for (const timer of timers) clearTimeout(timer); h.adapter.release.resolve(); await h.ctx.fiber.dispose() }
})

test('IO backlog retains each source and groups only requested steers before an explicit queue', async () => {
  const h = await harness(true)
  try {
    const done = idle(h.ctx, h.agent)
    const receive = (id, mode = 'auto') => h.io.receive({ bindingId: 'binding', externalMessageId: id, mode, content: [{ type: 'text', text: id }] })
    const first = await receive('start')
    await h.adapter.started.promise
    const a = await receive('steer-1')
    const b = await receive('steer-2')
    const c = await receive('steer-3')
    const queued = await receive('explicit-queue', 'queue')
    const last = await receive('after-queue')
    assert.deepEqual([first.mode, a.mode, b.mode, c.mode, queued.mode, last.mode], ['queue', 'steer', 'queue', 'queue', 'queue', 'queue'])
    h.adapter.release.resolve()
    await done; await h.io.drain()
    assert.equal(h.adapter.requests.length, 4)
    const admitted = inputs(h.agent).filter(item => item.source?.kind === 'super-agent-io')
    assert.deepEqual(admitted.map(item => item.source.externalMessageId), ['start', 'steer-1', 'steer-2', 'steer-3', 'explicit-queue', 'after-queue'])
    assert.deepEqual(admitted.map(item => item.id), [first.messageId, a.messageId, b.messageId, c.messageId, queued.messageId, last.messageId])
    const events = h.agent.session.snapshotEvents()
    const stepStarts = events.filter(event => event.type === 'step/start')
    const locations = admitted.map(input => {
      const event = events.find(event => event.type === 'user/message' && event.data.id === input.id)
      return stepStarts.findLast(step => step.seq < event.seq).data
    })
    assert.deepEqual(locations.slice(1, 4), [locations[1], locations[1], locations[1]])
    assert.notEqual(locations[4].turn, locations[1].turn)
    assert.notEqual(locations[5].turn, locations[4].turn)
    assert.equal(h.agent.inbox.nextTurn.length, 0)
    assert.equal(h.deliveries.length, 4)
    assert.ok([a, b, c].every(receipt => h.io.receipt(receipt.traceId).messageCommitted))
  } finally { h.adapter.release.resolve(); await h.ctx.fiber.dispose() }
})

test('local and mixed-channel steering keeps arrival order in one model step', async () => {
  const h = await harness(true)
  try {
    await h.io.putBinding({ id: 'other-binding', transport: 'mock-only', endpoint: 'mock:other', sessionId: h.agent.id, enabled: true, sendEnabled: true })
    const done = idle(h.ctx, h.agent)
    h.agent.followup(message('start'))
    await h.adapter.started.promise
    const receive = (bindingId, id, mode = 'steer') => h.io.receive({ bindingId, externalMessageId: id, mode, content: [{ type: 'text', text: id }] })
    const first = await receive('binding', 'first-external')
    const second = await receive('other-binding', 'second-external')
    h.agent.steer(message('third-local'))
    const fourth = await receive('binding', 'fourth-external')
    assert.deepEqual([first.mode, second.mode, fourth.mode], ['steer', 'queue', 'queue'])
    h.adapter.release.resolve()
    await done; await h.io.drain()
    assert.equal(h.adapter.requests.length, 2)
    const batch = h.adapter.requests[1].messages.filter(item => [first.messageId, second.messageId, 'third-local', fourth.messageId].includes(item.id))
    assert.deepEqual(batch.map(item => item.id), [first.messageId, second.messageId, 'third-local', fourth.messageId])
    assert.deepEqual(inputs(h.agent).slice(-4).map(item => item.id), batch.map(item => item.id))
  } finally { h.adapter.release.resolve(); await h.ctx.fiber.dispose() }
})

test('steers moved behind an explicit queued message join across channels when that queue reaches them', async () => {
  const h = await harness(true)
  try {
    await h.io.putBinding({ id: 'other-binding', transport: 'mock-only', endpoint: 'mock:other', sessionId: h.agent.id, enabled: true, sendEnabled: true })
    const done = idle(h.ctx, h.agent)
    h.agent.followup(message('start'))
    await h.adapter.started.promise
    const receive = (bindingId, id, mode) => h.io.receive({ bindingId, externalMessageId: id, mode, content: [{ type: 'text', text: id }] })
    const barrier = await receive('binding', 'explicit-queue', 'queue')
    const first = await receive('binding', 'queued-steer-1', 'steer')
    const second = await receive('other-binding', 'queued-steer-2', 'steer')
    assert.deepEqual([barrier.mode, first.mode, second.mode], ['queue', 'queue', 'queue'])
    h.adapter.release.resolve()
    await done; await h.io.drain()
    assert.equal(h.adapter.requests.length, 3)
    const batch = h.adapter.requests[2].messages.filter(item => [first.messageId, second.messageId].includes(item.id))
    assert.deepEqual(batch.map(item => item.id), [first.messageId, second.messageId])
  } finally { h.adapter.release.resolve(); await h.ctx.fiber.dispose() }
})

test('automatically routed steers behind a queued turn collect later arrivals from other channels', async () => {
  const h = await harness(true), timers = []
  try {
    await h.io.putBinding({ id: 'other-binding', transport: 'mock-only', endpoint: 'mock:other', sessionId: h.agent.id, enabled: true, sendEnabled: true })
    const done = idle(h.ctx, h.agent)
    h.agent.followup(message('start'))
    await h.adapter.started.promise
    const receive = (bindingId, id, mode) => h.io.receive({ bindingId, externalMessageId: id, mode, content: [{ type: 'text', text: id }] })
    await receive('binding', 'explicit-queue', 'queue')
    const first = await receive('binding', 'auto-steer-1', 'auto')
    const second = await receive('other-binding', 'auto-steer-2', 'auto')
    let late
    const scheduled = deferred()
    const off = h.ctx.on('agent/inbox/claimed', ({ agent, message: item }) => {
      if (agent !== h.agent || item.id !== first.messageId || late) return
      late = true
      timers.push(setTimeout(async () => { try { scheduled.resolve(await receive('other-binding', 'auto-steer-3', 'auto')) } catch (error) { scheduled.resolve({ error }) } }, 95))
    })
    h.adapter.release.resolve()
    const third = await scheduled.promise
    assert.ok(third.messageId, String(third.error ?? 'late message was not admitted'))
    await done; await h.io.drain()
    off()
    assert.equal(h.adapter.requests.length, 3)
    const batch = h.adapter.requests[2].messages.filter(item => [first.messageId, second.messageId, third.messageId].includes(item.id))
    assert.deepEqual(batch.map(item => item.id), [first.messageId, second.messageId, third.messageId])
  } finally { for (const timer of timers) clearTimeout(timer); h.adapter.release.resolve(); await h.ctx.fiber.dispose() }
})

test('unloading the plugin releases its pending wait and leaves subsequent messages to the loop', async () => {
  const h = await harness()
  try {
    let disposed = false
    const off = h.ctx.on('agent/inbox/claimed', ({ agent, message: item }) => {
      if (agent === h.agent && item.id === 'interrupt' && !disposed) {
        disposed = true
        setTimeout(() => { void h.fiber.dispose() }, 20)
      }
    })
    const done = idle(h.ctx, h.agent)
    h.agent.followup(message('start'))
    await h.adapter.started.promise
    h.agent.steer(message('interrupt'))
    h.adapter.release.resolve()
    await done
    off()
    assert.equal(disposed, true)
    assert.deepEqual(inputs(h.agent).map(item => item.id), ['start', 'interrupt'])
    assert.equal(h.agent.inbox.nextStep.length, 0)
  } finally { h.adapter.release.resolve(); await h.ctx.fiber.dispose() }
})
