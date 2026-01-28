import { Resolver, Query, Args, ID, ObjectType, Field, Int, InputType } from '@nestjs/graphql';
import { UseGuards } from '@nestjs/common';
import { VehiclesService } from './vehicles.service';
import { Vehicle } from './entities/vehicle.entity';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { CurrentUser } from '../auth/decorators/current-user.decorator';
import { User } from '../auth/entities/user.entity';

@ObjectType()
export class PaginatedVehicles {
    @Field(() => [Vehicle])
    items: Vehicle[];

    @Field(() => Int)
    total: number;

    @Field(() => Int)
    page: number;

    @Field(() => Int)
    limit: number;

    @Field(() => Int)
    totalPages: number;
}

@InputType()
export class VehicleSearchInput {
    @Field({ nullable: true })
    customerId?: string;

    @Field({ nullable: true })
    make?: string;

    @Field({ nullable: true })
    model?: string;

    @Field(() => Int, { nullable: true })
    year?: number;

    @Field({ nullable: true })
    vin?: string;

    @Field({ nullable: true })
    licensePlate?: string;

    @Field(() => Int, { nullable: true })
    page?: number;

    @Field(() => Int, { nullable: true })
    limit?: number;
}

@Resolver(() => Vehicle)
@UseGuards(JwtAuthGuard)
export class VehiclesResolver {
    constructor(private readonly vehiclesService: VehiclesService) { }

    @Query(() => PaginatedVehicles, { name: 'vehicles' })
    async findAll(
        @CurrentUser() user: User,
        @Args('options', { type: () => VehicleSearchInput, nullable: true }) options?: VehicleSearchInput,
    ): Promise<PaginatedVehicles> {
        return this.vehiclesService.findAll({
            tenantId: user.tenantId,
            ...options,
        });
    }

    @Query(() => Vehicle, { name: 'vehicle', nullable: true })
    async findOne(
        @CurrentUser() user: User,
        @Args('id', { type: () => ID }) id: string
    ): Promise<Vehicle> {
        return this.vehiclesService.findOne(user.tenantId, id);
    }
}
