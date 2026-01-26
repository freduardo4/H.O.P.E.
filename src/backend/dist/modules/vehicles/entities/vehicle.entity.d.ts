export declare enum FuelType {
    GASOLINE = "gasoline",
    DIESEL = "diesel",
    HYBRID = "hybrid",
    ELECTRIC = "electric",
    LPG = "lpg"
}
export declare enum TransmissionType {
    MANUAL = "manual",
    AUTOMATIC = "automatic",
    DSG = "dsg",
    CVT = "cvt"
}
export declare class Vehicle {
    id: string;
    tenantId: string;
    customerId: string;
    vin: string;
    make: string;
    model: string;
    year: number;
    variant: string;
    engineCode: string;
    engineDisplacement: number;
    enginePower: number;
    fuelType: FuelType;
    transmission: TransmissionType;
    licensePlate: string;
    mileage: number;
    ecuType: string;
    ecuSoftwareVersion: string;
    notes: string;
    isActive: boolean;
    createdAt: Date;
    updatedAt: Date;
    get displayName(): string;
}
