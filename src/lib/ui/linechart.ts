import * as d3 from 'd3';

type DataPoint = {
  x: number;
  y: number;
};

/**
 * A multi-series line chart that allows you to append new data points
 * as data becomes available.
 */
export class AppendingLineChart {
  private numLines: number;
  private data: DataPoint[] = [];
  private svg;
  private xScale;
  private yScale;
  private paths;
  private lineColors: string[];

  private minY = Number.MAX_VALUE;
  private maxY = Number.MIN_VALUE;

  constructor(container: d3.Selection<HTMLElement, unknown, HTMLElement, any>, lineColors: string[]) {
    this.lineColors = lineColors;
    this.numLines = lineColors.length;
    let node = container.node() as HTMLElement;
    let totalWidth = node.offsetWidth;
    let totalHeight = node.offsetHeight;
    let margin = {top: 2, right: 0, bottom: 2, left: 2};
    let width = totalWidth - margin.left - margin.right;
    let height = totalHeight - margin.top - margin.bottom;
    console.log(container.node());
    this.xScale = d3.scaleLinear()
      .domain([0, 0])
      .range([0, width]);

    this.yScale = d3.scaleLinear()
      .domain([0, 0])
      .range([height, 0]);
    this.svg = container.append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .attr("bgcolor", "blue")
      .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    this.paths = new Array(this.numLines);
    for (let i = 0; i < this.numLines; i++) {
      this.paths[i] = this.svg.append("path")
        .attr("class", "line")
        .attr("fill", "none")
        .attr("stroke", lineColors[i])
        .attr("stroke-width", "1.5px");
    }
  }

  reset() {
    this.data = [];
    this.redraw();
    this.minY = Number.MAX_VALUE;
    this.maxY = Number.MIN_VALUE;
  }

  addDataPoint(dataPoint: number) {
    this.minY = Math.min(this.minY, dataPoint);
    this.maxY = Math.max(this.maxY, dataPoint);

    this.data.push({x: this.data.length + 1, y: dataPoint});
    this.redraw();
  }

  private redraw() {
    // Adjust the x and y domain.
    this.xScale.domain([1, this.data.length]);
    this.yScale.domain([this.minY, this.maxY]);
    // Adjust all the <path> elements (lines).
    let getPathMap = (lineIndex: number) => {
      return d3.line<{x: number, y:number}>()
      .x(d => this.xScale(d.x))
      .y(d => this.yScale(d.y));
    };
    for (let i = 0; i < this.numLines; i++) {
      this.paths[i].datum(this.data).attr("d", getPathMap(i));
    }
  }
}