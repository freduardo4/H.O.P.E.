import { Injectable } from '@nestjs/common';

@Injectable()
export class ConfigService {
    private readonly flags = {
        enableBetaFlashing: false,
        enableExperimentalAI: true,
        minRequiredVoltage: 12.5,
        maintenanceMode: false,
        legalVersion: '1.0.1',
    };

    getFlags() {
        return this.flags;
    }

    getFlag(key: keyof typeof this.flags) {
        return this.flags[key];
    }
}
