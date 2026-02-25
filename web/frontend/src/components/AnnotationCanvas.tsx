import { useState, useEffect, useRef, useCallback } from 'react';
import { Stage, Layer, Image as KonvaImage, Rect, Circle, Text } from 'react-konva';
import { BoundingBox } from '../types';

interface AnnotationCanvasProps {
  imageUrl: string;
  bboxes: BoundingBox[];
  onBboxCreate: (bbox: Omit<BoundingBox, 'id' | 'color'>) => void;
  onBboxUpdate: (id: string, bbox: Partial<BoundingBox>) => void;
  onBboxDelete: (id: string) => void;
  selectedLabel: string | null;
  canvasWidth: number;
  canvasHeight: number;
  theme?: 'dark' | 'light';
}

type Interaction =
  | { type: 'drawing'; x1: number; y1: number; x2: number; y2: number }
  | { type: 'resizing'; bboxId: string; corner: string }
  | { type: 'moving'; bboxId: string; lastX: number; lastY: number };

export function AnnotationCanvas({
  imageUrl,
  bboxes,
  onBboxCreate,
  onBboxUpdate,
  selectedLabel,
  canvasWidth,
  canvasHeight,
  theme = 'dark',
}: AnnotationCanvasProps) {
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [scale, setScale] = useState(1);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [selectedBboxId, setSelectedBboxId] = useState<string | null>(null);
  const [interaction, setInteractionState] = useState<Interaction | null>(null);

  // Refs so event handlers always read the latest values without stale closures.
  const bboxesRef = useRef(bboxes);
  const scaleRef = useRef(scale);
  const interactionRef = useRef<Interaction | null>(null);
  const selectedLabelRef = useRef(selectedLabel);

  useEffect(() => { bboxesRef.current = bboxes; }, [bboxes]);
  useEffect(() => { scaleRef.current = scale; }, [scale]);
  useEffect(() => { selectedLabelRef.current = selectedLabel; }, [selectedLabel]);

  const setInteraction = useCallback((val: Interaction | null) => {
    interactionRef.current = val;
    setInteractionState(val);
  }, []);

  useEffect(() => {
    const img = new window.Image();
    img.src = imageUrl;
    img.onload = () => {
      setImage(img);
      const s = Math.min(canvasWidth / img.width, canvasHeight / img.height, 1);
      setScale(s);
      scaleRef.current = s;
      setImageSize({ width: img.width * s, height: img.height * s });
    };
  }, [imageUrl, canvasWidth, canvasHeight]);

  const getPos = (e: any) => e.target.getStage().getPointerPosition() as { x: number; y: number } | null;

  // Stage background mousedown: only fires if no child shape consumed the event.
  // Starts a new drawing interaction (if a label is selected) and deselects any box.
  const handleStageMouseDown = useCallback((e: any) => {
    // Ignore clicks that originated on a child shape (rect / handle).
    if (e.target !== e.target.getStage()) return;

    setSelectedBboxId(null);
    if (!selectedLabelRef.current) return;

    const pos = getPos(e);
    if (!pos) return;
    setInteraction({ type: 'drawing', x1: pos.x, y1: pos.y, x2: pos.x, y2: pos.y });
  }, [setInteraction]);

  // All mouse movement handled here — covers drawing, resizing, and moving.
  const handleStageMouseMove = useCallback((e: any) => {
    const ix = interactionRef.current;
    if (!ix) return;

    const pos = getPos(e);
    if (!pos) return;
    const s = scaleRef.current;

    if (ix.type === 'drawing') {
      setInteraction({ ...ix, x2: pos.x, y2: pos.y });
    } else if (ix.type === 'resizing') {
      const bbox = bboxesRef.current.find(b => b.id === ix.bboxId);
      if (!bbox) return;

      // Work in canvas-space coords, then convert back.
      let { x1, y1, x2, y2 } = {
        x1: bbox.x1 * s, y1: bbox.y1 * s,
        x2: bbox.x2 * s, y2: bbox.y2 * s,
      };
      switch (ix.corner) {
        case 'tl': x1 = pos.x; y1 = pos.y; break;
        case 'tr': x2 = pos.x; y1 = pos.y; break;
        case 'bl': x1 = pos.x; y2 = pos.y; break;
        case 'br': x2 = pos.x; y2 = pos.y; break;
        case 't':  y1 = pos.y; break;
        case 'b':  y2 = pos.y; break;
        case 'l':  x1 = pos.x; break;
        case 'r':  x2 = pos.x; break;
      }
      onBboxUpdate(ix.bboxId, {
        x1: Math.round(Math.min(x1, x2) / s),
        y1: Math.round(Math.min(y1, y2) / s),
        x2: Math.round(Math.max(x1, x2) / s),
        y2: Math.round(Math.max(y1, y2) / s),
      });
    } else if (ix.type === 'moving') {
      const bbox = bboxesRef.current.find(b => b.id === ix.bboxId);
      if (!bbox) return;

      const dx = Math.round((pos.x - ix.lastX) / s);
      const dy = Math.round((pos.y - ix.lastY) / s);

      if (dx !== 0 || dy !== 0) {
        onBboxUpdate(ix.bboxId, {
          x1: bbox.x1 + dx,
          y1: bbox.y1 + dy,
          x2: bbox.x2 + dx,
          y2: bbox.y2 + dy,
        });
        // Update lastX/Y so the next move frame computes a fresh delta.
        setInteraction({ ...ix, lastX: pos.x, lastY: pos.y });
      }
    }
  }, [setInteraction, onBboxUpdate]);

  // Finalise drawing on mouse up.
  const handleStageMouseUp = useCallback(() => {
    const ix = interactionRef.current;
    if (ix?.type === 'drawing') {
      const s = scaleRef.current;
      const label = selectedLabelRef.current;
      if (label) {
        const x1c = Math.min(ix.x1, ix.x2);
        const y1c = Math.min(ix.y1, ix.y2);
        const x2c = Math.max(ix.x1, ix.x2);
        const y2c = Math.max(ix.y1, ix.y2);
        if (x2c - x1c > 5 && y2c - y1c > 5) {
          onBboxCreate({
            x1: Math.round(x1c / s),
            y1: Math.round(y1c / s),
            x2: Math.round(x2c / s),
            y2: Math.round(y2c / s),
            label,
          });
        }
      }
    }
    setInteraction(null);
  }, [setInteraction, onBboxCreate]);

  // Bbox rect clicked → select it and start a move interaction.
  const handleRectMouseDown = useCallback((e: any, bboxId: string) => {
    e.cancelBubble = true; // prevent Stage from seeing this as a draw-start
    setSelectedBboxId(bboxId);
    const pos = getPos(e);
    if (!pos) return;
    setInteraction({ type: 'moving', bboxId, lastX: pos.x, lastY: pos.y });
  }, [setInteraction]);

  // Handle (corner / edge) mousedown → start a resize interaction.
  const handleHandleMouseDown = useCallback((e: any, bboxId: string, corner: string) => {
    e.cancelBubble = true;
    setInteraction({ type: 'resizing', bboxId, corner });
  }, [setInteraction]);

  const renderBbox = (bbox: BoundingBox) => {
    const s = scale;
    const x = bbox.x1 * s;
    const y = bbox.y1 * s;
    const w = (bbox.x2 - bbox.x1) * s;
    const h = (bbox.y2 - bbox.y1) * s;
    const isSelected = selectedBboxId === bbox.id;

    const corners = [
      { corner: 'tl', cx: x,     cy: y,     cursor: 'nwse-resize' },
      { corner: 'tr', cx: x + w, cy: y,     cursor: 'nesw-resize' },
      { corner: 'bl', cx: x,     cy: y + h, cursor: 'nesw-resize' },
      { corner: 'br', cx: x + w, cy: y + h, cursor: 'nwse-resize' },
    ];
    const edges = [
      { corner: 't', cx: x + w / 2, cy: y,         cursor: 'ns-resize' },
      { corner: 'b', cx: x + w / 2, cy: y + h,     cursor: 'ns-resize' },
      { corner: 'l', cx: x,         cy: y + h / 2, cursor: 'ew-resize' },
      { corner: 'r', cx: x + w,     cy: y + h / 2, cursor: 'ew-resize' },
    ];

    return (
      <>
        <Rect
          key={`rect-${bbox.id}`}
          x={x} y={y} width={w} height={h}
          stroke={bbox.color}
          strokeWidth={isSelected ? 3 : 2}
          fill={isSelected ? bbox.color + '20' : 'transparent'}
          onMouseDown={(e) => handleRectMouseDown(e, bbox.id)}
          style={{ cursor: 'move' }}
        />
        <Text
          key={`label-${bbox.id}`}
          x={x}
          y={Math.max(0, y - 20)}
          text={`${bbox.label}${bbox.score !== undefined ? ` ${(bbox.score * 100).toFixed(0)}%` : ''}`}
          fontSize={13}
          fill={bbox.color}
          fontStyle="bold"
          listening={false}
        />
        {isSelected && corners.map(({ corner, cx, cy, cursor }) => (
          <Circle
            key={`${bbox.id}-${corner}`}
            x={cx} y={cy}
            radius={6}
            fill="white"
            stroke={bbox.color}
            strokeWidth={2}
            onMouseDown={(e) => handleHandleMouseDown(e, bbox.id, corner)}
            style={{ cursor }}
          />
        ))}
        {isSelected && edges.map(({ corner, cx, cy, cursor }) => (
          <Circle
            key={`${bbox.id}-${corner}`}
            x={cx} y={cy}
            radius={5}
            fill="white"
            stroke={bbox.color}
            strokeWidth={2}
            onMouseDown={(e) => handleHandleMouseDown(e, bbox.id, corner)}
            style={{ cursor }}
          />
        ))}
      </>
    );
  };

  // In-progress draw preview box.
  const drawPreview = interaction?.type === 'drawing' ? (
    <Rect
      x={Math.min(interaction.x1, interaction.x2)}
      y={Math.min(interaction.y1, interaction.y2)}
      width={Math.abs(interaction.x2 - interaction.x1)}
      height={Math.abs(interaction.y2 - interaction.y1)}
      stroke="#22c55e"
      strokeWidth={2}
      dash={[5, 5]}
      listening={false}
    />
  ) : null;

  return (
    <div className={`border-2 ${theme === 'dark' ? 'border-slate-700 bg-slate-900' : 'border-gray-300 bg-white'} rounded-lg overflow-hidden`}>
      <Stage
        width={canvasWidth}
        height={canvasHeight}
        onMouseDown={handleStageMouseDown}
        onMouseMove={handleStageMouseMove}
        onMouseUp={handleStageMouseUp}
      >
        <Layer>
          {image && <KonvaImage image={image} width={imageSize.width} height={imageSize.height} />}
          {bboxes.map(renderBbox)}
          {drawPreview}
        </Layer>
      </Stage>
    </div>
  );
}
