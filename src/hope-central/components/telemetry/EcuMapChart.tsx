'use client';

import React from 'react';
import ReactECharts from 'echarts-for-react';
import 'echarts-gl'; // Required for 3D charts

interface EcuMapChartProps {
    title: string;
    data: number[][]; // [x, y, z] format
}

export default function EcuMapChart({ title, data }: EcuMapChartProps) {
    const option = {
        title: {
            text: title,
            textStyle: { color: '#fff' }
        },
        tooltip: {},
        backgroundColor: 'transparent',
        visualMap: {
            show: true,
            dimension: 2,
            min: 0,
            max: 20, // Example AFR range
            inRange: {
                color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
            },
            textStyle: { color: '#fff' }
        },
        xAxis3D: {
            type: 'value',
            name: 'RPM',
            nameTextStyle: { color: '#fff' },
            axisLabel: { color: '#ccc' }
        },
        yAxis3D: {
            type: 'value',
            name: 'Load',
            nameTextStyle: { color: '#fff' },
            axisLabel: { color: '#ccc' }
        },
        zAxis3D: {
            type: 'value',
            name: 'AFR',
            nameTextStyle: { color: '#fff' },
            axisLabel: { color: '#ccc' }
        },
        grid3D: {
            viewControl: {
                projection: 'perspective'
            },
            axisLine: {
                lineStyle: { color: '#fff' }
            }
        },
        series: [{
            type: 'surface',
            wireframe: {
                show: true
            },
            data: data
        }]
    };

    return (
        <ReactECharts
            option={option}
            style={{ height: '400px', width: '100%' }}
            theme="dark"
        />
    );
}
