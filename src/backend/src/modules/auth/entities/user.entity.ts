import {
    Entity,
    PrimaryGeneratedColumn,
    Column,
    CreateDateColumn,
    UpdateDateColumn,
} from 'typeorm';
import { ObjectType, Field, ID, registerEnumType } from '@nestjs/graphql';

export enum UserRole {
    ADMIN = 'admin',
    SHOP_OWNER = 'shop_owner',
    TECHNICIAN = 'technician',
    VIEWER = 'viewer',
}

registerEnumType(UserRole, {
    name: 'UserRole',
});

@ObjectType()
@Entity('users')
export class User {
    @Field(() => ID)
    @PrimaryGeneratedColumn('uuid')
    id: string;

    @Field()
    @Column({ unique: true })
    email: string;

    @Column()
    passwordHash: string;

    @Field()
    @Column()
    firstName: string;

    @Field()
    @Column()
    lastName: string;

    @Field(() => UserRole)
    @Column({ type: 'simple-enum', enum: UserRole, default: UserRole.TECHNICIAN })
    role: UserRole;

    @Field({ nullable: true })
    @Column({ nullable: true })
    tenantId: string;

    @Field()
    @Column({ default: true })
    isActive: boolean;

    @Column({ nullable: true })
    refreshToken: string;

    @Field({ nullable: true })
    @Column({ type: 'datetime', nullable: true })
    lastLoginAt: Date;

    @Field()
    @Column({ default: '1.0.0' })
    acceptedLegalVersion: string;

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
}
