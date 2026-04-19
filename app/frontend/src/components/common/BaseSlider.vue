<template>
  <div class="base-slider">
    <div class="base-slider__header">
      <span class="base-slider__label">{{ label }}</span>
      <span class="base-slider__value">{{ displayValue }}</span>
    </div>

    <input
      class="base-slider__input"
      type="range"
      :min="min"
      :max="max"
      :step="step"
      :value="modelValue"
      @input="onInput"
    />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  label: string
  modelValue: number
  min?: number
  max?: number
  step?: number
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: number): void
}>()

function onInput(e: Event) {
  const value = Number((e.target as HTMLInputElement).value)
  emit('update:modelValue', value)
}

const displayValue = computed(() => {
  return Number(props.modelValue).toFixed(props.step && props.step < 1 ? 2 : 0)
})
</script>

<style scoped>
.base-slider {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.base-slider__header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.base-slider__label {
  font-size: 13px;
  color: var(--text-secondary);
}

.base-slider__value {
  font-size: 12px;
  color: var(--text-primary);
  padding: 4px 8px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.06);
}

.base-slider__input {
  appearance: none;
  width: 100%;
  height: 6px;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--accent-soft), rgba(255, 255, 255, 0.08));
  outline: none;
}

.base-slider__input::-webkit-slider-thumb {
  appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 999px;
  background: #fff;
  box-shadow: 0 0 0 4px rgba(124, 140, 255, 0.18);
  cursor: pointer;
}
</style>
