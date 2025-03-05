"use client";

import { ColumnDef } from "@tanstack/react-table";
import { Customer, User } from "@prisma/client";
import { Button } from "@/components/ui/button";
import { MoreHorizontal } from "lucide-react";

import Link from "next/link";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from "../ui/dropdown-menu";

export type CustomerWithAssignedTo = Customer & {
  assignedTo: User | null;
  predictions: {
    churnRisk: number;
    createdAt: Date;
  }[];
};

export const columns: ColumnDef<CustomerWithAssignedTo>[] = [
  {
    accessorKey: "name",
    header: "Name",
  },
  {
    accessorKey: "email",
    header: "Email",
  },
  {
    accessorKey: "phone",
    header: "Phone",
  },
  {
    accessorKey: "assignedTo.name",
    header: "Assigned To",
    cell: ({ row }) => row.original.assignedTo?.name || "Unassigned",
  },
  {
    accessorKey: "predictions",
    header: "Churn Risk",
    cell: ({ row }) => {
      const latestPrediction = row.original.predictions[0];
      if (!latestPrediction) return "No data";
      const risk = (latestPrediction.churnRisk * 100).toFixed(1);
      return `${risk}%`;
    },
  },
  {
    id: "actions",
    cell: ({ row }) => {
      const customer = row.original;

      return (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="h-8 w-8 p-0">
              <span className="sr-only">Open menu</span>
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>Actions</DropdownMenuLabel>
            <DropdownMenuItem asChild>
              <Link href={`/dashboard/customers/${customer.id}`}>
                View details
              </Link>
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem asChild>
              <Link href={`/dashboard/customers/${customer.id}/edit`}>
                Edit customer
              </Link>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      );
    },
  },
]; 