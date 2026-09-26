import assert from 'node:assert/strict'
import test from 'node:test'
import { switchTextProvider } from '../src/lib/productionProfile'

const minimax = { provider: 'minimax', model: 'MiniMax-M3', base_url: 'https://api.minimax.io' }

test('switching from MiniMax to Ollama drops the MiniMax model and URL', () => {
  assert.deepEqual(switchTextProvider(minimax, 'ollama'), { provider: 'ollama', model: '', base_url: 'http://127.0.0.1:11434' })
})

test('switching to local clears hosted leftovers; a custom Ollama URL survives', () => {
  assert.deepEqual(switchTextProvider(minimax, 'local'), { provider: 'local', model: '', base_url: '' })
  const ollama = { provider: 'ollama', model: 'llama3.1', base_url: 'http://192.168.1.5:11434' }
  assert.deepEqual(switchTextProvider(ollama, 'ollama'), ollama)
  assert.equal(switchTextProvider(ollama, 'minimax').model, 'MiniMax-M3')
})
