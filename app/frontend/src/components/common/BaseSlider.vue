<template>
  <div class="base-slider">
    <div class="base-slider__header">
      <span 
        class="base-slider__label" 
        title="双击重置"
        style="cursor: pointer; user-select: none;"
        @dblclick="reset"
      >
        {{ label }}
      </span>
      <span 
        class="base-slider__value" 
        style="cursor: pointer;" 
        @click="reset"
      >
        {{ displayValue }}
      </span>
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

const props = withDefaults(defineProps<{
  label: string
  modelValue: number
  min?: number
  max?: number
  step?: number
  // 3. 新增默认值属性，默认为 0
  defaultValue?: number 
}>(), {
  min: -100,
  max: 100,
  step: 1,
  defaultValue: 0
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: number): void
}>()

function onInput(e: Event) {
  const value = Number((e.target as HTMLInputElement).value)
  emit('update:modelValue', value)
}

function reset() {
  emit('update:modelValue', props.defaultValue)
}

const displayValue = computed(() => {
  const val = Number(props.modelValue || 0)
  return val.toFixed(props.step && props.step < 1 ? 2 : 0)
})
</script>

<style scoped>
.base-slider {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.base-slider__label {
  font-size: 13px;
  color: var(--text-secondary);
  transition: color 0.2s;
}

/* 如果数值不是默认值，可以给 label 一个高亮色（可选） */
.base-slider:has(input:not([value="0"])) .base-slider__label {
  color: var(--accent-color);
}
</style>
