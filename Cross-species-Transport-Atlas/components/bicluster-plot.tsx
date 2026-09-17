'use client';

import Image from 'next/image';

export function BiclusterPlot({asset}:{asset:string}){
  return <div className="reference-bicluster-shell">
    <Image
      className="reference-bicluster-image"
      src={asset}
      alt="Block-diagonal bicluster heatmap with category-coloured mouse and human gene labels"
      width={1800}
      height={1450}
      unoptimized
      priority
    />
  </div>;
}
