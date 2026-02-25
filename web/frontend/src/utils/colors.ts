export const COLOR_SCHEMES = {
  vibrant: {
    name: 'Vibrant',
    colors: [
      '#ef4444', // red
      '#3b82f6', // blue
      '#eab308', // yellow
      '#ec4899', // pink
      '#06b6d4', // cyan
      '#10b981', // green
      '#8b5cf6', // purple
      '#f97316', // orange
    ]
  },
  pastel: {
    name: 'Pastel',
    colors: [
      '#fca5a5', // light red
      '#93c5fd', // light blue
      '#fde047', // light yellow
      '#f9a8d4', // light pink
      '#67e8f9', // light cyan
      '#6ee7b7', // light green
      '#c4b5fd', // light purple
      '#fdba74', // light orange
    ]
  },
  neon: {
    name: 'Neon',
    colors: [
      '#ff0080', // hot pink
      '#00ffff', // cyan
      '#ff00ff', // magenta
      '#00ff00', // lime
      '#ffff00', // yellow
      '#ff6600', // orange
      '#0099ff', // sky blue
      '#cc00ff', // purple
    ]
  },
  earth: {
    name: 'Earth Tones',
    colors: [
      '#a16207', // amber
      '#65a30d', // lime
      '#0891b2', // cyan
      '#c026d3', // fuchsia
      '#dc2626', // red
      '#ea580c', // orange
      '#059669', // emerald
      '#7c3aed', // violet
    ]
  },
  monochrome: {
    name: 'Monochrome',
    colors: [
      '#ef4444', // red
      '#f97316', // orange
      '#eab308', // yellow
      '#84cc16', // lime
      '#10b981', // green
      '#06b6d4', // cyan
      '#3b82f6', // blue
      '#8b5cf6', // purple
    ]
  },
  highContrast: {
    name: 'High Contrast',
    colors: [
      '#ff0000', // pure red
      '#0000ff', // pure blue
      '#ffff00', // pure yellow
      '#ff00ff', // pure magenta
      '#00ffff', // pure cyan
      '#00ff00', // pure green
      '#ff8800', // pure orange
      '#8800ff', // pure purple
    ]
  }
};

export type ColorScheme = keyof typeof COLOR_SCHEMES;

export const getColorForIndex = (index: number, scheme: ColorScheme = 'vibrant'): string => {
  const colors = COLOR_SCHEMES[scheme].colors;
  return colors[index % colors.length];
};
