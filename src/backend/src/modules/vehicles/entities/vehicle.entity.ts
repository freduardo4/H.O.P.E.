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
import { ObjectType, Field, ID, Int, Float, registerEnumType } from '@nestjs/graphql';

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

registerEnumType(FuelType, {
    name: 'FuelType',
});

registerEnumType(TransmissionType, {
    name: 'TransmissionType',
});

@ObjectType()
@Entity('vehicles')
export class Vehicle {
    @Field(() => ID)
    @PrimaryGeneratedColumn('uuid')
    id: string;

    @Field()
    @Column()
    tenantId: string;

    @Field({ nullable: true })
    @Column({ nullable: true })
    customerId: string;

    @Field({ nullable: true })
    @Column({ length: 17, nullable: true })
    vin: string;

    @Field()
    @Column()
    make: string;

    @Field()
    @Column()
    model: string;

    @Field(() => Int)
    @Column()
    year: number;

    @Field({ nullable: true })
    @Column({ nullable: true })
    variant: string;

    @Field({ nullable: true })
    @Column({ nullable: true })
    engineCode: string;

    @Field(() => Float, { nullable: true })
    @Column({ nullable: true })
    engineDisplacement: number;

    @Field(() => Int, { nullable: true })
    @Column({ nullable: true })
    enginePower: number;

    @Field(() => FuelType, { nullable: true })
    @Column({ type: 'enum', enum: FuelType, nullable: true })
    fuelType: FuelType;

    @Field(() => TransmissionType, { nullable: true })
    @Column({ type: 'enum', enum: TransmissionType, nullable: true })
    transmission: TransmissionType;

    @Field({ nullable: true })
    @Column({ nullable: true })
    licensePlate: string;

    @Field(() => Int, { nullable: true })
    @Column({ nullable: true })
    mileage: number;

    @Field({ nullable: true })
    @Column({ nullable: true })
    ecuType: string;

    @Field({ nullable: true })
    @Column({ nullable: true })
    ecuSoftwareVersion: string;

    @Field({ nullable: true })
    @Column({ nullable: true })
    notes: string;

    @Field()
    @Column({ default: true })
    isActive: boolean;

    @Field()
    @CreateDateColumn()
    createdAt: Date;

    @Field()
    @UpdateDateColumn()
    updatedAt: Date;

    @Field()
    get displayName(): string {
        return `${this.year} ${this.make} ${this.model}${this.variant ? ` ${this.variant}` : ''}`;
    }
}
