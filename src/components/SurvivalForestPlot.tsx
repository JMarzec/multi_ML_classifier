const HIGH_RISK_COLOR = "hsl(var(--destructive))";
const LOW_RISK_COLOR = "hsl(var(--success))";

// Forest Plot SVG component with proper box-plot style markers
interface ForestPlotEntry {
  gene: string;
  hr: number;
  hr_lower: number;
  hr_upper: number;
  p_value: number;
  significant: boolean;
}

export const ForestPlotSVG = ({ data }: { data: ForestPlotEntry[] }) => {
  const width = 900;
  const rowHeight = 22;
  const height = Math.max(400, data.length * rowHeight + 100);
  const margin = { top: 40, right: 180, left: 140, bottom: 60 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  // Calculate x-axis domain
  const allValues = data.flatMap((d) => [d.hr_lower, d.hr_upper, d.hr]);
  const minVal = Math.min(...allValues, 0.1);
  const maxVal = Math.max(...allValues, 2);
  const xMin = Math.max(0.01, minVal * 0.8);
  const xMax = maxVal * 1.2;

  // Use log scale for HR visualization (common in forest plots)
  const logScale = (value: number) => {
    const logMin = Math.log10(xMin);
    const logMax = Math.log10(xMax);
    const logVal = Math.log10(Math.max(value, xMin));
    return ((logVal - logMin) / (logMax - logMin)) * plotWidth;
  };

  // Y position for each gene
  const yPosition = (index: number) => {
    return (index + 0.5) * (plotHeight / data.length);
  };

  // Generate x-axis ticks (log scale friendly values)
  const generateTicks = () => {
    const ticks: number[] = [];
    const logMin = Math.log10(xMin);
    const logMax = Math.log10(xMax);
    
    // Add nice round values
    [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50].forEach((v) => {
      if (v >= xMin && v <= xMax) ticks.push(v);
    });
    
    return ticks;
  };

  const xTicks = generateTicks();
  const hrOnePosition = logScale(1);

  return (
    <svg 
      viewBox={`0 0 ${width} ${height}`} 
      className="w-full max-w-4xl mx-auto"
      style={{ minHeight: height }}
    >
      {/* Background */}
      <rect width={width} height={height} fill="hsl(var(--card))" rx={8} />

      {/* Plot area */}
      <g transform={`translate(${margin.left}, ${margin.top})`}>
        {/* Grid lines */}
        {xTicks.map((tick) => (
          <line
            key={tick}
            x1={logScale(tick)}
            y1={0}
            x2={logScale(tick)}
            y2={plotHeight}
            stroke="hsl(var(--border))"
            strokeDasharray={tick === 1 ? "none" : "3 3"}
            strokeWidth={tick === 1 ? 2 : 1}
            strokeOpacity={tick === 1 ? 0.8 : 0.4}
          />
        ))}

        {/* Horizontal grid lines for each gene */}
        {data.map((_, i) => (
          <line
            key={i}
            x1={0}
            y1={yPosition(i)}
            x2={plotWidth}
            y2={yPosition(i)}
            stroke="hsl(var(--border))"
            strokeOpacity={0.2}
          />
        ))}

        {/* HR = 1 reference line (emphasized) */}
        <line
          x1={hrOnePosition}
          y1={-10}
          x2={hrOnePosition}
          y2={plotHeight + 10}
          stroke="hsl(var(--muted-foreground))"
          strokeWidth={2}
          strokeDasharray="8 4"
        />
        <text
          x={hrOnePosition}
          y={-20}
          textAnchor="middle"
          className="text-xs fill-muted-foreground font-medium"
        >
          HR = 1
        </text>

        {/* Forest plot entries */}
        {data.map((entry, i) => {
          const y = yPosition(i);
          const hrX = logScale(entry.hr);
          const lowerX = logScale(entry.hr_lower);
          const upperX = logScale(entry.hr_upper);
          const color = entry.hr > 1 ? HIGH_RISK_COLOR : LOW_RISK_COLOR;
          const opacity = entry.significant ? 1 : 0.5;

          return (
            <g key={entry.gene} opacity={opacity}>
              {/* Gene label */}
              <text
                x={-10}
                y={y}
                textAnchor="end"
                alignmentBaseline="middle"
                className="text-xs fill-foreground font-mono"
              >
                {entry.gene.length > 18 ? entry.gene.slice(0, 16) + "..." : entry.gene}
              </text>

              {/* Confidence interval line */}
              <line
                x1={lowerX}
                y1={y}
                x2={upperX}
                y2={y}
                stroke={color}
                strokeWidth={2}
              />

              {/* CI caps */}
              <line
                x1={lowerX}
                y1={y - 5}
                x2={lowerX}
                y2={y + 5}
                stroke={color}
                strokeWidth={2}
              />
              <line
                x1={upperX}
                y1={y - 5}
                x2={upperX}
                y2={y + 5}
                stroke={color}
                strokeWidth={2}
              />

              {/* Point estimate (diamond/square marker) */}
              <rect
                x={hrX - 5}
                y={y - 5}
                width={10}
                height={10}
                fill={color}
                stroke="hsl(var(--background))"
                strokeWidth={1}
                transform={`rotate(45, ${hrX}, ${y})`}
              />

              {/* HR value label on right */}
              <text
                x={plotWidth + 10}
                y={y}
                textAnchor="start"
                alignmentBaseline="middle"
                className="text-xs fill-muted-foreground font-mono"
              >
                {entry.hr.toFixed(2)} ({entry.hr_lower.toFixed(2)}-{entry.hr_upper.toFixed(2)})
              </text>
            </g>
          );
        })}

        {/* X-axis */}
        <g transform={`translate(0, ${plotHeight})`}>
          <line x1={0} y1={0} x2={plotWidth} y2={0} stroke="hsl(var(--muted-foreground))" />
          {xTicks.map((tick) => (
            <g key={tick} transform={`translate(${logScale(tick)}, 0)`}>
              <line y1={0} y2={6} stroke="hsl(var(--muted-foreground))" />
              <text
                y={20}
                textAnchor="middle"
                className="text-xs fill-muted-foreground"
              >
                {tick}
              </text>
            </g>
          ))}
        </g>

        {/* Labels for risk interpretation - positioned above axis label */}
        <text
          x={hrOnePosition / 2}
          y={plotHeight + 25}
          textAnchor="middle"
          className="text-xs fill-success font-medium"
        >
          ← Protective
        </text>
        <text
          x={hrOnePosition + (plotWidth - hrOnePosition) / 2}
          y={plotHeight + 25}
          textAnchor="middle"
          className="text-xs fill-destructive font-medium"
        >
          Risk →
        </text>

        {/* X-axis label - positioned below risk labels */}
        <text
          x={plotWidth / 2}
          y={plotHeight + 50}
          textAnchor="middle"
          className="text-sm fill-muted-foreground font-medium"
        >
          Hazard Ratio (log scale)
        </text>
      </g>

      {/* Column header for HR values */}
      <text
        x={margin.left + plotWidth + 10}
        y={margin.top - 15}
        textAnchor="start"
        className="text-xs fill-muted-foreground font-medium"
      >
        HR (95% CI)
      </text>
    </svg>
  );
};

