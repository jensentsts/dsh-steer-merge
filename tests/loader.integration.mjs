import test from 'node:test'
import assert from 'node:assert/strict'
import { Context, FiberState } from '@deepseek-ai/cordis'
import Loader from '@deepseek-ai/cordis-plugin-loader'
import Include from '@deepseek-ai/cordis-plugin-include'

test('official loader activates and removes the steer-merge entry', async () => {
  const ctx = new Context()
  try {
    await ctx.plugin(Loader)
    ctx.loader.builtins.include = Include
    await ctx.loader.create({ name: 'cordis:include', config: { path: new URL('./fixtures/cordis.yml', import.meta.url).href } })
    await ctx.loader.await()
    const entries = [...ctx.loader.entries()]
    for (const id of ['sessions', 'projections', 'agents', 'dsh-steer-merge']) {
      const entry = entries.find(item => item.options.id === id)
      assert.equal(entry?.fiber?.state, FiberState.ACTIVE, id + ': ' + (entry?.error?.message ?? ''))
    }
    const entry = entries.find(item => item.options.id === 'dsh-steer-merge')
    entry.parent.remove(entry.options.id)
    await entry.fiber.inertia
    assert.ok(ctx.get('agents'))
  } finally { await ctx.fiber.dispose() }
})
