using System;
using System.Drawing;
using TUIO;

public class TuioDemoObject : TuioObject
{
    private SolidBrush black = new SolidBrush(Color.Black);
    private SolidBrush white = new SolidBrush(Color.White);

    public TuioDemoObject(long s_id, int f_id, float xpos, float ypos, float angle)
        : base(s_id, f_id, xpos, ypos, angle)
    {
    }

    public TuioDemoObject(TuioObject o)
        : base(o)
    {
    }

    public void paint(Graphics g)
    {
        int Xpos = (int)(xpos * SmartKitchenDemo.width);
        int Ypos = (int)(ypos * SmartKitchenDemo.height);
        int size = SmartKitchenDemo.height / 10;

        g.TranslateTransform(Xpos, Ypos);
        g.RotateTransform((float)(angle / Math.PI * 180.0f));
        g.TranslateTransform(-Xpos, -Ypos);

        g.FillRectangle(black, new Rectangle(Xpos - size / 2, Ypos - size / 2, size, size));

        g.TranslateTransform(Xpos, Ypos);
        g.RotateTransform(-1 * (float)(angle / Math.PI * 180.0f));
        g.TranslateTransform(-Xpos, -Ypos);

        Font font = new Font("Arial", 10.0f);
        g.DrawString(symbol_id + "", font, white, new PointF(Xpos - 10, Ypos - 10));
    }
}