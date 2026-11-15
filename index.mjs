import z from '@deepseek-ai/schemastery'

export const name = 'dsh-steer-merge'
export const inject = ['agents']
export const Config = z.object({
  enabled: z.boolean().default(true),
  windowMs: z.number().step(1).min(0).max(2000).default(500),
  maxWaitMs: z.number().step(1).min(0).max(10000).default(2000),
}).volatile()

const nativeContext = new Set(['runtime-context', 'time-context', 'agent-instructions', 'skill-catalog', 'system-prompt', 'session-reference', 'repeat-tool-reminder', 'working-directory', 'model-selection'])
const external = message => message?.source?.kind === 'super-agent-io' && message.source.version === 1

/** Wait only at a boundary that already contains steering; release when stopped. */
function pause(milliseconds, signal, stopped) {
  return new Promise((resolve, reject) => {
    let timer
    const finish = (error) => {
      clearTimeout(timer)
      signal.removeEventListener('abort', onAbort)
      stopped.removeEventListener('abort', onStop)
      if (error) reject(error); else resolve()
    }
    const onAbort = () => finish(signal.reason ?? new Error('Turn stopped'))
    const onStop = () => finish()
    if (signal.aborted) return onAbort()
    if (stopped.aborted || milliseconds <= 0) return onStop()
    signal.addEventListener('abort', onAbort, { once: true })
    stopped.addEventListener('abort', onStop, { once: true })
    timer = setTimeout(onStop, milliseconds)
  })
}

/** Collect a burst until it has been quiet, with a finite maximum wait. */
async function collectBurst(windowMs, maxWaitMs, state, signal, stopped) {
  if (windowMs <= 0) return
  const started = performance.now(), limit = started + Math.max(windowMs, maxWaitMs)
  let observed = state.sequence, deadline = started + windowMs
  while (true) {
    await pause(Math.max(0, deadline - performance.now()), signal, stopped)
    signal.throwIfAborted()
    if (stopped.aborted || state.sequence === observed) return
    observed = state.sequence
    deadline = Math.min(limit, Math.max(started, state.changedAt) + windowMs)
    if (deadline <= performance.now()) return
  }
}

/** Resolve whether IO originally wanted steering before it fell back to a queued turn. */
function wantedSteer(io, message, bindingId, explicitOnly = false) {
  if (!io || !external(message) || message.source.bindingId !== bindingId) return false
  let receipt
  try { receipt = io.receipt(message.source.traceId) } catch { return false }
  return receipt?.messageId === message.id && receipt.bindingId === bindingId && receipt.mode === message.source.inputMode
    && (receipt.requestedMode === 'steer' || !explicitOnly && receipt.requestedMode === 'auto')
}

/**
 * Use the exact-release inbox's claim operation so Session splices and claimed
 * notifications remain owned by the native loop. If a future driver lacks
 * that operation, it retains its original behavior without changing the inbox.
 */
export function apply(ctx, config) {
  ctx.effect(() => {
    const stopped = new AbortController()
    const steers = new WeakMap()
    const bucket = agent => {
      let value = steers.get(agent)
      if (!value) { value = { pending: new Set(), claimed: new Set(), queuedSteers: new Set(), claimedQueued: new Set(), order: new Map(), sequence: 0, changedAt: 0 }; steers.set(agent, value) }
      return value
    }
    const off = [
      ctx.on('agent/inbox/inserted', ({ agent, message }) => {
        const state = bucket(agent)
        state.order.set(message.id, ++state.sequence)
        state.changedAt = performance.now()
        if (agent.inbox?.nextStep?.some(item => item.id === message.id)) state.pending.add(message.id)
        else if (agent.status === 'running' && agent.inbox?.nextTurn?.some(item => item.id === message.id) && wantedSteer(ctx.get('superAgentIo'), message, message.source?.bindingId)) state.queuedSteers.add(message.id)
      }),
      ctx.on('agent/inbox/claimed', ({ agent, message }) => {
        const state = steers.get(agent)
        if (state?.pending.delete(message.id)) state.claimed.add(message.id)
        if (state?.queuedSteers.delete(message.id)) state.claimedQueued.add(message.id)
      }),
      ctx.on('agent/inbox/discarded', ({ agent, message }) => {
        const state = steers.get(agent)
        state?.pending.delete(message.id); state?.claimed.delete(message.id); state?.queuedSteers.delete(message.id); state?.claimedQueued.delete(message.id); state?.order.delete(message.id)
      }),
      ctx.on('agent/disposed', ({ agent }) => steers.delete(agent)),
      ctx.on('agent/pre-step', async ({ agent, messages, turn, signal }, next) => {
        const state = bucket(agent)
        const batch = new Set()
        for (const message of messages) {
          const claimedSteer = state.claimed.delete(message.id)
          const queuedSteer = state.claimedQueued.delete(message.id)
          if ((claimedSteer || queuedSteer) && !nativeContext.has(message.source?.kind)) batch.add(message.id)
        }
        const hasSteer = batch.size > 0
        try {
          if (!config.get().enabled || typeof agent.inbox?.claim !== 'function') return next()
          if (hasSteer) await collectBurst(config.get().windowMs, config.get().maxWaitMs ?? 2000, state, signal, stopped.signal)
          signal.throwIfAborted()
          if (stopped.signal.aborted || ctx.agents.get(agent.id) !== agent || !config.get().enabled) return next()
          const io = ctx.get('superAgentIo')
          const claim = target => {
            const claimed = agent.inbox.claim(target, turn)
            for (const message of claimed) {
              const wasSteer = state.claimed.delete(message.id) && !nativeContext.has(message.source?.kind)
              const wasQueuedSteer = state.claimedQueued.delete(message.id)
              if (wasSteer || wasQueuedSteer || target === 'next-turn' && wantedSteer(io, message, message.source?.bindingId)) batch.add(message.id)
            }
            messages.push(...claimed)
          }

          // Use the native inbox so each claimed message retains its source and attachment.
          if (hasSteer && agent.inbox.nextStep.length) claim('next-step')

          // The IO fairness gate can move requested steers into next-turn. Once
          // this boundary has an actual steer, gather its contiguous backlog
          // from every channel; an explicitly queued message ends the batch.
          const input = messages.filter(message => !nativeContext.has(message.source?.kind))
          const bindingId = input.length && input.every(external) ? input[0].source.bindingId : undefined
          const sameBinding = bindingId && input.every(message => message.source.bindingId === bindingId)
          const eligibleInput = sameBinding && input.every(message => wantedSteer(io, message, bindingId))
          const mixedBindings = hasSteer || eligibleInput && input.every(message => wantedSteer(io, message, bindingId, true))
          if (eligibleInput) for (const message of input) batch.add(message.id)
          if (typeof io?.receipt === 'function' && (hasSteer || eligibleInput)) {
            for (const message of [...agent.inbox.nextTurn]) {
              if (agent.inbox.nextTurn[0]?.id !== message.id || message.source?.inputMode !== 'queue' ||
                !wantedSteer(io, message, message.source?.bindingId) || !mixedBindings && message.source?.bindingId !== bindingId) break
              claim('next-turn')
            }
          }

          // Native next-step and next-turn lists can interleave arrivals. Sort
          // only the claimed steering messages, leaving other context in place.
          const positions = messages.flatMap((message, index) => batch.has(message.id) ? [index] : [])
          if (positions.every(index => state.order.has(messages[index].id))) {
            const ordered = positions.map(index => messages[index]).sort((a, b) => state.order.get(a.id) - state.order.get(b.id))
            positions.forEach((index, offset) => { messages[index] = ordered[offset] })
          }
          return next()
        } finally {
          for (const message of messages) state.order.delete(message.id)
        }
      }),
    ]
    return () => { stopped.abort(); for (const remove of off.reverse()) remove() }
  })
}
