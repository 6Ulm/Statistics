'use client';

import {useEffect,useRef} from 'react';
import type {ECharts,EChartsOption} from 'echarts';

export function EChart({option,className,preserveDataZoom=false}:{option:EChartsOption;className?:string;preserveDataZoom?:boolean}){
  const element=useRef<HTMLDivElement>(null);
  const chartRef=useRef<ECharts|null>(null);
  useEffect(()=>{
    let disposed=false;
    let observer:ResizeObserver|undefined;
    void import('echarts').then(echarts=>{
      if(disposed||!element.current)return;
      chartRef.current=echarts.init(element.current,undefined,{renderer:'canvas'});
      observer=new ResizeObserver(()=>chartRef.current?.resize());
      observer.observe(element.current);
    });
    return()=>{disposed=true;observer?.disconnect();chartRef.current?.dispose();chartRef.current=null};
  },[]);
  useEffect(()=>{
    let cancelled=false;
    void import('echarts').then(echarts=>{
      if(cancelled||!element.current)return;
      const chart=echarts.getInstanceByDom(element.current);
      if(chart){
        chartRef.current=chart;
        const previousZoom=preserveDataZoom?(chart.getOption().dataZoom as Array<{start?:number;end?:number}>|undefined)?.map(({start,end})=>({start,end})):undefined;
        chart.setOption(option,{notMerge:true});
        previousZoom?.forEach((zoom,index)=>chart.dispatchAction({type:'dataZoom',dataZoomIndex:index,...zoom}));
      }
    });
    return()=>{cancelled=true};
  },[option,preserveDataZoom]);
  return <div ref={element} className={className}/>;
}
