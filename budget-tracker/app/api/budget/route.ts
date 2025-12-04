import { NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { startOfMonth, endOfMonth } from 'date-fns';

export async function GET() {
  const now = new Date();
  const month = now.getMonth() + 1;
  const year = now.getFullYear();

  const monthStart = startOfMonth(now);
  const monthEnd = endOfMonth(now);

  const budgetItems = await prisma.budgetItem.findMany({
    where: { month, year },
    include: { category: true },
  });

  // Calculate spent amounts per category
  const result = await Promise.all(
    budgetItems.map(async (item) => {
      const spent = await prisma.transaction.aggregate({
        where: {
          categoryId: item.categoryId,
          date: { gte: monthStart, lte: monthEnd },
          amount: { gt: 0 },
        },
        _sum: { amount: true },
      });

      return {
        ...item,
        spent: spent._sum.amount || 0,
        remaining: item.monthlyAmount - (spent._sum.amount || 0),
      };
    })
  );

  return NextResponse.json(result);
}

export async function PUT(request: Request) {
  const { itemId, monthlyAmount } = await request.json();

  const updated = await prisma.budgetItem.update({
    where: { id: itemId },
    data: {
      monthlyAmount,
      weeklyAmount: monthlyAmount / 4.33,
    },
  });

  return NextResponse.json(updated);
}
