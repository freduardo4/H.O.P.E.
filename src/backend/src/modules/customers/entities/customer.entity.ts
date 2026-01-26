import {
    Entity,
    PrimaryGeneratedColumn,
    Column,
    CreateDateColumn,
    UpdateDateColumn,
} from 'typeorm';

export enum CustomerType {
    INDIVIDUAL = 'individual',
    BUSINESS = 'business',
    FLEET = 'fleet',
}

@Entity('customers')
export class Customer {
    @PrimaryGeneratedColumn('uuid')
    id: string;

    @Column()
    tenantId: string;

    @Column({ type: 'enum', enum: CustomerType, default: CustomerType.INDIVIDUAL })
    type: CustomerType;

    @Column()
    firstName: string;

    @Column()
    lastName: string;

    @Column({ nullable: true })
    companyName: string;

    @Column({ unique: true })
    email: string;

    @Column({ nullable: true })
    phone: string;

    @Column({ nullable: true })
    alternatePhone: string;

    @Column({ nullable: true })
    address: string;

    @Column({ nullable: true })
    city: string;

    @Column({ nullable: true })
    state: string;

    @Column({ nullable: true })
    postalCode: string;

    @Column({ nullable: true })
    country: string;

    @Column({ nullable: true })
    taxId: string;

    @Column({ nullable: true })
    notes: string;

    @Column('simple-json', { nullable: true })
    preferences: {
        contactMethod?: 'email' | 'phone' | 'sms';
        receiveMarketing?: boolean;
        preferredLanguage?: string;
    };

    @Column({ default: true })
    isActive: boolean;

    @CreateDateColumn()
    createdAt: Date;

    @UpdateDateColumn()
    updatedAt: Date;

    get fullName(): string {
        return `${this.firstName} ${this.lastName}`;
    }

    get displayName(): string {
        return this.companyName || this.fullName;
    }
}
