<template>
  <section class="panel">
    <h3>导出</h3>

    <!-- 胶片边框水印选项 -->
    <div class="frame-option">
      <label class="toggle-row">
        <span class="toggle-label">胶片边框水印</span>
        <span
          class="toggle-switch"
          :class="{ active: editor.film.frame_border?.enabled }"
          @click="toggleBorder"
        >
          <span class="toggle-knob"></span>
        </span>
      </label>
      <p class="frame-hint">在图片两侧叠加黑色齿孔与黄色文字（胶卷名、用户、序号）</p>
    </div>

    <button :disabled="!editor.resultUrl" @click="downloadImage">下载结果</button>
  </section>
</template>

<script setup lang="ts">
import { useEditorStore } from '@/stores/editorStore';

const editor = useEditorStore();

function toggleBorder() {
  if (!editor.film.frame_border) return;
  editor.film.frame_border.enabled = !editor.film.frame_border.enabled;
}

/** 渲染胶片边框水印到 Canvas 上 */
function renderFrameBorder(
  ctx: CanvasRenderingContext2D,
  img: HTMLImageElement,
  filmName: string,
  imageIndex: number,
) {
  const w = img.naturalWidth;
  const h = img.naturalHeight;

  ctx.canvas.width = w;
  ctx.canvas.height = h;

  // 1. 绘制原图
  ctx.drawImage(img, 0, 0);

  // 2. 全图微弱半透明黑色遮罩
  ctx.fillStyle = 'rgba(0, 0, 0, 0.18)';
  ctx.fillRect(0, 0, w, h);

  // 3. 基础尺寸与参数计算
  const stripW = Math.round(w * 0.07);                     // 边缘区域参考宽度
  const sprocketW = Math.round(stripW * 0.48);             // 齿孔宽度
  const sprocketH = Math.round(sprocketW * 0.65);          // 齿孔高度
  const sprocketGap = Math.round(sprocketH * 2.3);         // 齿孔间距
  const radius = Math.max(2, Math.round(sprocketH * 0.25));  // 齿孔圆角
  const totalCycle = sprocketH + sprocketGap;

  // 4. 齿孔位置计算（靠内侧）
  const sprocketMarginInner = Math.round(stripW * 0.45);
  const leftSprocketX = sprocketMarginInner;
  const rightSprocketX = w - sprocketMarginInner - sprocketW;

  // 5. 绘制上下完全铺满的齿孔
  const drawSprocketsColumn = (startX: number) => {
    ctx.fillStyle = '#000000';
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1;

    const count = Math.ceil(h / totalCycle) + 1;
    const startY = (h - (count * totalCycle - sprocketGap)) / 2;

    for (let i = 0; i < count; i++) {
      const y = startY + i * totalCycle;
      if (y + sprocketH > 0 && y < h) {
        ctx.beginPath();
        ctx.roundRect(startX, y, sprocketW, sprocketH, radius);
        ctx.fill();
        ctx.stroke();
      }
    }
  };

  drawSprocketsColumn(leftSprocketX);
  drawSprocketsColumn(rightSprocketX);

  // 6. 绘制外侧文字的基础设置
  ctx.fillStyle = '#e5be53'; // 胶片复古黄色
  const textEdgeGap = Math.round(stripW * 0.3); // 基础留白

  /** 顺时针旋转 90 度绘制文本 */
  const drawRotatedText = (text: string, x: number, y: number, align: 'left' | 'right') => {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(Math.PI / 2);
    ctx.textAlign = align === 'left' ? 'left' : 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 0, 0);
    ctx.restore();
  };

  // --- 【左侧文字设置】 ---
  const leftFontSize = Math.max(11, Math.round(stripW * 0.22));
  ctx.font = `600 ${leftFontSize}px "Courier New", Consolas, monospace`;

  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, '0');
  const day = String(now.getDate()).padStart(2, '0');
  const hours = String(now.getHours()).padStart(2, '0');
  const minutes = String(now.getMinutes()).padStart(2, '0');
  const dateStr = `${year}.${month}.${day} ${hours}:${minutes}`;

  const leftTextMarginOuter = Math.round(stripW * 0.22);
  const leftTextX = leftTextMarginOuter;

  drawRotatedText(dateStr, leftTextX, textEdgeGap, 'left');
  drawRotatedText('DigitalFilm', leftTextX, h - textEdgeGap, 'right');

  // --- ⚡ 【右侧文字设置：字体放大 & 向中间靠拢】 ---
  // 1. 进一步放大字体
  const rightFontSize = Math.max(15, Math.round(stripW * 0.36)); 
  ctx.font = `bold ${rightFontSize}px "Courier New", Consolas, monospace`;

  // 2. 控制 X 坐标靠近齿孔外侧
  const rightTextMarginOuter = Math.round(stripW * 0.35); 
  const rightTextX = w - rightTextMarginOuter;

  // 3. 增加 Y 轴偏移量（让上方名称向下走，下方序号向上走）
  const moveCenterOffsetY = Math.round(stripW * 1.5); // 向中间靠拢的距离，可按需微调
  const rightTopY = textEdgeGap + moveCenterOffsetY;          // 向下移
  const rightBottomY = h - textEdgeGap - moveCenterOffsetY;   // 向上移

  const deviceStr = filmName || 'Xiaomi 13 Ultra';
  const indexStr = `${imageIndex} ►`;

  drawRotatedText(deviceStr, rightTextX, rightTopY, 'left');
  drawRotatedText(indexStr, rightTextX, rightBottomY, 'right');
}

async function downloadImage() {
  const img = document.querySelector<HTMLImageElement>('img.preview-image');
  if (!img) {
    console.warn('未找到目标图片元素');
    return;
  }
  const src = img.src;
  if (!src) return;

  const border = editor.film.frame_border;
  const needsBorder = border?.enabled;

  if (!needsBorder) {
    const a = document.createElement('a');
    a.href = src;
    a.download = 'film-result.png';
    a.click();
    return;
  }

  // 带水印：用 Canvas 合成后下载
  const image = new Image();
  image.crossOrigin = 'anonymous';
  image.onload = () => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;

    const imageIndex = border?.image_index ?? 1;
    const displayName = border?.film_name ?? 'FILM';

    renderFrameBorder(ctx, image, displayName, imageIndex);

    const a = document.createElement('a');
    a.href = canvas.toDataURL('image/png');
    a.download = 'film-result-framed.png';
    a.click();
  };
  image.onerror = () => {
    const a = document.createElement('a');
    a.href = src;
    a.download = 'film-result.png';
    a.click();
  };
  image.src = src;
}
</script>

<style scoped>
.panel {
  padding: 0;
}

.panel h3 {
  margin: 0 0 12px;
  font-size: 13px;
  color: var(--text-secondary, #888);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.frame-option {
  margin-bottom: 16px;
  padding: 10px 12px;
  background: var(--bg-panel-hover, rgba(255, 255, 255, 0.04));
  border-radius: 8px;
}

.toggle-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  cursor: pointer;
  user-select: none;
}

.toggle-label {
  font-size: 13px;
  font-weight: 500;
  color: var(--text-primary, #ddd);
}

.toggle-switch {
  position: relative;
  width: 36px;
  height: 20px;
  background: var(--bg-toggle-off, #444);
  border-radius: 10px;
  transition: background 0.2s;
}

.toggle-switch.active {
  background: var(--accent-color, #e8c547);
}

.toggle-knob {
  position: absolute;
  top: 2px;
  left: 2px;
  width: 16px;
  height: 16px;
  background: #fff;
  border-radius: 50%;
  transition: transform 0.2s;
}

.toggle-switch.active .toggle-knob {
  transform: translateX(16px);
}

.frame-hint {
  margin: 6px 0 0;
  font-size: 11px;
  color: var(--text-tertiary, #777);
  line-height: 1.4;
}

button {
  width: 100%;
  padding: 10px 16px;
  border: none;
  border-radius: 8px;
  background: var(--accent-color, #e8c547);
  color: var(--bg-app, #1a1a1a);
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 0.2s;
}

button:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}

button:not(:disabled):hover {
  opacity: 0.9;
}
</style>
