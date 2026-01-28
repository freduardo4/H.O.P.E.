
import { Injectable } from '@nestjs/common';

@Injectable()
export class AfrService {
    /**
     * Generates a default Target AFR map for a naturally aspirated engine.
     * axis X: RPM (e.g. 16 breakpoints)
     * axis Y: Load/MAP (e.g. 16 breakpoints)
     */
    generateDefaultMap(rows: number = 16, cols: number = 16): number[][] {
        const map: number[][] = [];

        for (let r = 0; r < rows; r++) {
            const rowData: number[] = [];
            // Normalize load from 0.0 to 1.0 (0 to 100%)
            const load = r / (rows - 1);

            for (let c = 0; c < cols; c++) {
                // Normalize RPM from 0.0 to 1.0 (Idle to Redline)
                const rpm = c / (cols - 1);

                let targetAfr = 14.7;

                if (load > 0.8) {
                    // WOT / High Load enrichment
                    // Interpolate from 13.5 down to 12.5 at high RPM
                    targetAfr = 13.0 - (0.5 * rpm);
                } else if (load > 0.4) {
                    // Mid load - transition
                    targetAfr = 14.7 - (1.0 * (load - 0.4));
                } else {
                    // Low load / Cruise
                    targetAfr = 14.7;
                }

                // Round to 2 decimal places
                rowData.push(Math.round(targetAfr * 100) / 100);
            }
            map.push(rowData);
        }

        return map;
    }
}
