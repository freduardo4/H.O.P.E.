import {
    Entity,
    PrimaryGeneratedColumn,
    Column,
    CreateDateColumn,
    UpdateDateColumn,
    ManyToOne,
    JoinColumn,
} from 'typeorm';

export enum UserRole {
    ADMIN = 'admin',
    SHOP_OWNER = 'shop_owner',
    TECHNICIAN = 'technician',
    VIEWER = 'viewer',
}

@Entity('users')
export class User {
    @PrimaryGeneratedColumn('uuid')
    id: string;

    @Column({ unique: true })
    email: string;

    @Column()
    passwordHash: string;

    @Column()
    firstName: string;

    @Column()
    lastName: string;

    @Column({ type: 'enum', enum: UserRole, default: UserRole.TECHNICIAN })
    role: UserRole;

    @Column({ nullable: true })
    tenantId: string;

    @Column({ default: true })
    isActive: boolean;

    @Column({ nullable: true })
    refreshToken: string;

    @Column({ nullable: true })
    lastLoginAt: Date;

    @CreateDateColumn()
    createdAt: Date;

    @UpdateDateColumn()
    updatedAt: Date;

    get fullName(): string {
        return `${this.firstName} ${this.lastName}`;
    }
}
