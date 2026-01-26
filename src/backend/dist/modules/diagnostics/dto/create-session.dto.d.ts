import { SessionType } from '../entities/diagnostic-session.entity';
export declare class CreateSessionDto {
    vehicleId: string;
    type?: SessionType;
    mileageAtSession?: number;
    notes?: string;
}
