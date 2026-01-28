import { Entity, PrimaryGeneratedColumn, Column, CreateDateColumn, ManyToOne } from 'typeorm';
import { ObjectType, Field, ID } from '@nestjs/graphql';
import { CalibrationListing } from './calibration-listing.entity';
import { User } from '../../auth/entities/user.entity';

@ObjectType()
@Entity('licenses')
export class License {
    @Field(() => ID)
    @PrimaryGeneratedColumn('uuid')
    id: string;

    @Field()
    @Column()
    licenseKey: string;

    @Field()
    @Column()
    hardwareId: string; // The fingerprint (SHA-256)

    @Field(() => CalibrationListing)
    @ManyToOne(() => CalibrationListing, (listing) => listing.licenses)
    listing: CalibrationListing;

    @Field(() => User)
    @ManyToOne(() => User)
    user: User;

    @Field()
    @Column({ default: true })
    isActive: boolean;

    @Field()
    @CreateDateColumn()
    createdAt: Date;
}
