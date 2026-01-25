import {
    Entity,
    PrimaryGeneratedColumn,
    Column,
    CreateDateColumn,
    UpdateDateColumn,
    ManyToOne,
    OneToMany,
    JoinColumn,
} from 'typeorm';

export enum FuelType {
    GASOLINE = 'gasoline',
    DIESEL = 'diesel',
    HYBRID = 'hybrid',
    ELECTRIC = 'electric',
    LPG = 'lpg',
}

export enum TransmissionType {
    MANUAL = 'manual',
    AUTOMATIC = 'automatic',
    DSG = 'dsg',
    CVT = 'cvt',
}

@Entity('vehicles')
export class Vehicle {
    @PrimaryGeneratedColumn('uuid')
    id: string;

    @Column()
    tenantId: string;

    @Column({ nullable: true })
    customerId: string;

    @Column({ length: 17, nullable: true })
    vin: string;

    @Column()
    make: string;

    @Column()
    model: string;

    @Column()
    year: number;

    @Column({ nullable: true })
    variant: string;

    @Column({ nullable: true })
    engineCode: string;

    @Column({ nullable: true })
    engineDisplacement: number;

    @Column({ nullable: true })
    enginePower: number;

    @Column({ type: 'enum', enum: FuelType, nullable: true })
    fuelType: FuelType;

    @Column({ type: 'enum', enum: TransmissionType, nullable: true })
    transmission: TransmissionType;

    @Column({ nullable: true })
    licensePlate: string;

    @Column({ nullable: true })
    mileage: number;

    @Column({ nullable: true })
    ecuType: string;

    @Column({ nullable: true })
    ecuSoftwareVersion: string;

    @Column({ nullable: true })
    notes: string;

    @Column({ default: true })
    isActive: boolean;

    @CreateDateColumn()
    createdAt: Date;

    @UpdateDateColumn()
    updatedAt: Date;

    get displayName(): string {
        return `${this.year} ${this.make} ${this.model}${this.variant ? ` ${this.variant}` : ''}`;
    }
}
