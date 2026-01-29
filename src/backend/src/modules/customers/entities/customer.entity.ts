import {
    Entity,
    PrimaryGeneratedColumn,
    Column,
    CreateDateColumn,
    UpdateDateColumn,
} from 'typeorm';
import { ObjectType, Field, ID, registerEnumType } from '@nestjs/graphql';

export enum CustomerType {
    INDIVIDUAL = 'individual',
    BUSINESS = 'business',
    FLEET = 'fleet',
}

registerEnumType(CustomerType, { name: 'CustomerType' });

@ObjectType()
@Entity('customers')
export class Customer {
    @Field(() => ID)
    @PrimaryGeneratedColumn('uuid')
    id: string;

    @Field()
    @Column()
    tenantId: string;

    @Field(() => CustomerType)
    @Column({ type: 'simple-enum', enum: CustomerType, default: CustomerType.INDIVIDUAL })
    type: CustomerType;

    @Field()
    @Column()
    firstName: string;

    @Field()
    @Column()
    lastName: string;

    @Field({ nullable: true })
    @Column({ nullable: true })
    companyName: string;

    @Field()
    @Column({ unique: true })
    email: string;

    @Field({ nullable: true })
    @Column({ nullable: true })
    phone: string;

    @Field({ nullable: true })
    @Column({ nullable: true })
    alternatePhone: string;

    @Field({ nullable: true })
    @Column({ nullable: true })
    address: string;

    @Field({ nullable: true })
    @Column({ nullable: true })
    city: string;

    @Field({ nullable: true })
    @Column({ nullable: true })
    state: string;

    @Field({ nullable: true })
    @Column({ nullable: true })
    postalCode: string;

    @Field({ nullable: true })
    @Column({ nullable: true })
    country: string;

    @Field({ nullable: true })
    @Column({ nullable: true })
    taxId: string;

    @Field({ nullable: true })
    @Column({ nullable: true })
    notes: string;

    @Field(() => String, { nullable: true })
    @Column('simple-json', { nullable: true })
    preferences: {
        contactMethod?: 'email' | 'phone' | 'sms';
        receiveMarketing?: boolean;
        preferredLanguage?: string;
    };

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
    get fullName(): string {
        return `${this.firstName} ${this.lastName}`;
    }

    @Field()
    get displayName(): string {
        return this.companyName || this.fullName;
    }
}
