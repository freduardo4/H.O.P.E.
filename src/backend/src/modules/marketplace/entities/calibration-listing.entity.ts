import { Entity, PrimaryGeneratedColumn, Column, CreateDateColumn, UpdateDateColumn, OneToMany } from 'typeorm';
import { ObjectType, Field, ID, Float } from '@nestjs/graphql';
import { License } from './license.entity';

@ObjectType()
@Entity('calibration_listings')
export class CalibrationListing {
    @Field(() => ID)
    @PrimaryGeneratedColumn('uuid')
    id: string;

    @Field()
    @Column()
    title: string;

    @Field()
    @Column('text')
    description: string;

    @Field(() => Float)
    @Column('decimal', { precision: 10, scale: 2 })
    price: number;

    @Field()
    @Column()
    version: string;

    @Field()
    @Column()
    compatibility: string; // e.g., "EDC17C64", "MED17.5"

    @Field()
    @Column()
    fileUrl: string; // S3 or local path to encrypted binary

    @Field()
    @Column()
    checksum: string;

    @Field(() => [ID], { nullable: true })
    @OneToMany(() => License, (license) => license.listing)
    licenses: License[];

    @Field()
    @CreateDateColumn()
    createdAt: Date;

    @Field()
    @UpdateDateColumn()
    updatedAt: Date;
}
